//! MobileNetV4-style convolutional vision backbone.
//!
//! The implementation follows the MobileNetV4 *Universal Inverted Bottleneck*
//! (UIB) design: an optional start depthwise convolution, optional expansion,
//! a middle depthwise convolution, optional squeeze-excite, and a linear
//! projection, with residual connections when the stride is one and the
//! channels match. The published channel table is exposed as
//! [`MobileNetV4Config::conv_small`] and can be replaced by any table of
//! [`BlockSpec`] values.
//!
//! The hybrid MobileNetV4 variants add Mobile MQA attention; that path is not
//! implemented here (the conv-small family is self-contained and trains on CPU).

use oxide_ml_core::{Dim, Error, Result, Shape, ValueId};
use oxide_ml_model::{ModelContext, Module, ModuleList, ModulePath, ParameterRegistry};

use crate::modules::{BatchNorm, Conv2d, Linear};
use oxide_ml_graph::Conv2dConfig;

/// Elementwise activation used after a convolution/normalization pair.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConvActivation {
    #[default]
    Relu,
    Gelu,
    Identity,
}

/// Convolution + BatchNorm without an activation.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
pub struct ConvNorm {
    pub conv: Conv2d,
    pub bn: BatchNorm,
}

impl ConvNorm {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        groups: usize,
    ) -> Self {
        let padding = [kernel[0] / 2, kernel[1] / 2];
        let config = Conv2dConfig {
            stride,
            padding,
            dilation: [1, 1],
            groups,
        };
        ConvNorm {
            conv: Conv2d::with_config(in_channels, out_channels, kernel, config).without_bias(),
            bn: BatchNorm::new(out_channels),
        }
    }

    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    pub fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        self.bn.forward(self.conv.forward(x))
    }
}

/// Convolution + BatchNorm + activation.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
pub struct ConvNormAct {
    pub norm: ConvNorm,
    #[module(skip)]
    pub activation: ConvActivation,
}

impl ConvNormAct {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        groups: usize,
        activation: ConvActivation,
    ) -> Self {
        ConvNormAct {
            norm: ConvNorm::new(in_channels, out_channels, kernel, stride, groups),
            activation,
        }
    }

    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        let y = self.norm.bn.forward(self.norm.conv.forward(x));
        if self.activation == ConvActivation::Relu {
            y.relu()
        } else if self.activation == ConvActivation::Gelu {
            y.gelu()
        } else {
            y
        }
    }
}

/// Global average pool over `[N, C, H, W]` to `[N, C]`.
#[derive(Clone, Debug, Default)]
pub struct GlobalAvgPool;

impl Module for GlobalAvgPool {
    fn module_type(&self) -> &'static str {
        "GlobalAvgPool"
    }

    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let shape = cx.value_shape(input)?;
        if shape.rank() != 4 {
            return Err(Error::RankMismatch {
                context: "GlobalAvgPool".into(),
                expected: 4,
                got: shape.rank(),
            });
        }
        let height = shape
            .dim(2)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("GlobalAvgPool", "height must be static"))?;
        let width = shape
            .dim(3)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("GlobalAvgPool", "width must be static"))?;
        let pooled = cx.builder().avg_pool2d(
            input,
            oxide_ml_graph::Pool2dConfig {
                kernel: [height, width],
                stride: [1, 1],
                padding: [0, 0],
                dilation: [1, 1],
            },
        )?;
        let channels = shape
            .dim(1)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("GlobalAvgPool", "channels must be static"))?;
        Ok(vec![cx.builder().reshape(
            pooled,
            Shape::new(vec![
                shape.dim(0).unwrap_or(Dim::Dynamic),
                Dim::Static(channels),
            ]),
        )?])
    }
}

/// Squeeze-and-excitation gate over channels.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model, forward_multi = apply)]
pub struct SqueezeExcite {
    pub reduce: Linear,
    pub expand: Linear,
    #[module(skip)]
    pub channels: usize,
}

impl SqueezeExcite {
    pub fn new(channels: usize, reduction: usize) -> Self {
        let hidden = (channels / reduction).max(1);
        SqueezeExcite {
            reduce: Linear::with_seed(channels, hidden, true, 0x5EED_5EED),
            expand: Linear::with_seed(hidden, channels, true, 0x5EED_5EED + 1),
            channels,
        }
    }

    fn apply(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        let input = *inputs
            .first()
            .ok_or_else(|| Error::invalid_graph("SqueezeExcite needs one input"))?;
        let shape = cx.value_shape(input)?;
        if shape.rank() != 4 {
            return Err(Error::RankMismatch {
                context: "SqueezeExcite".into(),
                expected: 4,
                got: shape.rank(),
            });
        }
        let height = shape
            .dim(2)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("SqueezeExcite", "height must be static"))?;
        let width = shape
            .dim(3)
            .and_then(Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("SqueezeExcite", "width must be static"))?;
        let pooled = cx.builder().avg_pool2d(
            input,
            oxide_ml_graph::Pool2dConfig {
                kernel: [height, width],
                stride: [1, 1],
                padding: [0, 0],
                dilation: [1, 1],
            },
        )?;
        let batch = shape.dim(0).unwrap_or(Dim::Dynamic);
        let pooled = cx
            .builder()
            .reshape(pooled, Shape::new(vec![batch, Dim::Static(self.channels)]))?;
        let hidden = oxide_ml_model::module_forward(cx, "reduce", &self.reduce, pooled)?;
        let hidden = cx.builder().relu(hidden)?;
        let gate = oxide_ml_model::module_forward(cx, "expand", &self.expand, hidden)?;
        let gate = cx.builder().sigmoid(gate)?;
        // Broadcast the gate over the spatial dimensions.
        let gate = cx.builder().reshape(
            gate,
            Shape::new(vec![
                batch,
                Dim::Static(self.channels),
                Dim::Static(1),
                Dim::Static(1),
            ]),
        )?;
        let target = Shape::new(vec![
            shape.dim(0).unwrap_or(Dim::Dynamic),
            Dim::Static(self.channels),
            Dim::Static(height),
            Dim::Static(width),
        ]);
        let gate = cx.builder().broadcast_to(gate, target)?;
        Ok(vec![cx.builder().mul(input, gate)?])
    }
}

/// One UIB (Universal Inverted Bottleneck) block description.
#[derive(Clone, Copy, Debug)]
pub struct UibSpec {
    pub in_channels: usize,
    pub out_channels: usize,
    pub expand_channels: usize,
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub use_start_depthwise: bool,
    pub use_expand: bool,
    pub use_se: bool,
    pub activation: ConvActivation,
}

/// Universal Inverted Bottleneck block.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
pub struct UibBlock {
    pub start_dw: Option<ConvNormAct>,
    pub expand: Option<ConvNormAct>,
    pub middle_dw: ConvNormAct,
    pub se: Option<SqueezeExcite>,
    pub project: ConvNorm,
    #[module(skip)]
    pub residual: bool,
    #[module(skip)]
    pub has_start_dw: bool,
    #[module(skip)]
    pub has_expand: bool,
    #[module(skip)]
    pub has_se: bool,
}

impl UibBlock {
    pub fn new(spec: UibSpec) -> Self {
        let stride = spec.stride;
        let hidden = spec.expand_channels.max(spec.in_channels);
        let start_dw = spec.use_start_depthwise.then(|| {
            ConvNormAct::new(
                spec.in_channels,
                spec.in_channels,
                spec.kernel,
                stride,
                spec.in_channels,
                spec.activation,
            )
        });
        let middle_stride = if start_dw.is_some() { [1, 1] } else { stride };
        let expand = spec.use_expand.then(|| {
            ConvNormAct::new(spec.in_channels, hidden, [1, 1], [1, 1], 1, spec.activation)
        });
        let middle_dw = ConvNormAct::new(
            hidden,
            hidden,
            spec.kernel,
            middle_stride,
            hidden,
            spec.activation,
        );
        let se = spec.use_se.then(|| SqueezeExcite::new(hidden, 4));
        let project = ConvNorm::new(hidden, spec.out_channels, [1, 1], [1, 1], 1);
        let residual = stride == [1, 1] && spec.in_channels == spec.out_channels;
        UibBlock {
            has_start_dw: start_dw.is_some(),
            has_expand: expand.is_some(),
            has_se: se.is_some(),
            start_dw,
            expand,
            middle_dw,
            se,
            project,
            residual,
        }
    }

    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        let mut y = x.clone();
        if self.has_start_dw {
            y = self.start_dw.forward(y);
        }
        if self.has_expand {
            y = self.expand.forward(y);
        }
        let y = self.middle_dw.forward(y);
        let y = if self.has_se { self.se.forward(y) } else { y };
        let y = self.project.forward(y);
        if self.residual { x + y } else { y }
    }
}

/// A vision block: a plain convolution block or a UIB block.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
#[allow(clippy::large_enum_variant)]
pub enum MobileNetBlock {
    #[module(body)]
    Conv(ConvNormAct),
    #[module(body)]
    Uib(UibBlock),
}

/// One entry in a MobileNetV4 channel table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockSpec {
    /// Plain convolution + BN + activation.
    Conv {
        out_channels: usize,
        kernel: usize,
        stride: usize,
    },
    /// Universal Inverted Bottleneck.
    Uib {
        out_channels: usize,
        expand_ratio: usize,
        kernel: usize,
        stride: usize,
        start_depthwise: bool,
        se: bool,
    },
}

/// MobileNetV4 configuration.
#[derive(Clone, Debug)]
pub struct MobileNetV4Config {
    pub num_classes: usize,
    pub stem_channels: usize,
    pub head_channels: usize,
    pub blocks: Vec<BlockSpec>,
    pub activation: ConvActivation,
}

impl MobileNetV4Config {
    pub fn new(num_classes: usize) -> Self {
        Self::conv_small(num_classes)
    }

    /// A compact table for CPU training at small resolutions.
    pub fn nano(num_classes: usize) -> Self {
        MobileNetV4Config {
            num_classes,
            stem_channels: 16,
            head_channels: 96,
            blocks: vec![
                BlockSpec::Conv {
                    out_channels: 16,
                    kernel: 3,
                    stride: 1,
                },
                BlockSpec::Uib {
                    out_channels: 24,
                    expand_ratio: 2,
                    kernel: 3,
                    stride: 2,
                    start_depthwise: false,
                    se: false,
                },
                BlockSpec::Uib {
                    out_channels: 24,
                    expand_ratio: 2,
                    kernel: 3,
                    stride: 1,
                    start_depthwise: false,
                    se: false,
                },
                BlockSpec::Uib {
                    out_channels: 32,
                    expand_ratio: 2,
                    kernel: 5,
                    stride: 2,
                    start_depthwise: true,
                    se: false,
                },
                BlockSpec::Uib {
                    out_channels: 32,
                    expand_ratio: 2,
                    kernel: 5,
                    stride: 1,
                    start_depthwise: true,
                    se: true,
                },
                BlockSpec::Uib {
                    out_channels: 48,
                    expand_ratio: 3,
                    kernel: 3,
                    stride: 2,
                    start_depthwise: true,
                    se: false,
                },
                BlockSpec::Uib {
                    out_channels: 48,
                    expand_ratio: 3,
                    kernel: 3,
                    stride: 1,
                    start_depthwise: true,
                    se: true,
                },
                BlockSpec::Uib {
                    out_channels: 64,
                    expand_ratio: 3,
                    kernel: 5,
                    stride: 2,
                    start_depthwise: true,
                    se: false,
                },
                BlockSpec::Uib {
                    out_channels: 64,
                    expand_ratio: 3,
                    kernel: 5,
                    stride: 1,
                    start_depthwise: true,
                    se: true,
                },
            ],
            activation: ConvActivation::Relu,
        }
    }

    /// MobileNetV4-Conv-Small-style channel table (width 1.0).
    pub fn conv_small(num_classes: usize) -> Self {
        MobileNetV4Config {
            num_classes,
            stem_channels: 32,
            head_channels: 256,
            blocks: vec![
                BlockSpec::Conv {
                    out_channels: 32,
                    kernel: 3,
                    stride: 1,
                },
                BlockSpec::Uib {
                    out_channels: 64,
                    expand_ratio: 2,
                    kernel: 3,
                    stride: 2,
                    start_depthwise: true,
                    se: false,
                },
                BlockSpec::Uib {
                    out_channels: 64,
                    expand_ratio: 2,
                    kernel: 3,
                    stride: 1,
                    start_depthwise: true,
                    se: true,
                },
                BlockSpec::Uib {
                    out_channels: 96,
                    expand_ratio: 3,
                    kernel: 3,
                    stride: 2,
                    start_depthwise: true,
                    se: false,
                },
                BlockSpec::Uib {
                    out_channels: 96,
                    expand_ratio: 3,
                    kernel: 5,
                    stride: 1,
                    start_depthwise: true,
                    se: true,
                },
                BlockSpec::Uib {
                    out_channels: 128,
                    expand_ratio: 4,
                    kernel: 3,
                    stride: 2,
                    start_depthwise: true,
                    se: false,
                },
                BlockSpec::Uib {
                    out_channels: 128,
                    expand_ratio: 4,
                    kernel: 5,
                    stride: 1,
                    start_depthwise: true,
                    se: true,
                },
                BlockSpec::Uib {
                    out_channels: 160,
                    expand_ratio: 4,
                    kernel: 3,
                    stride: 1,
                    start_depthwise: true,
                    se: true,
                },
            ],
            activation: ConvActivation::Relu,
        }
    }

    /// Deepest channel in the table (used for validation).
    pub fn channels(&self) -> usize {
        self.blocks
            .last()
            .map(|block| match block {
                BlockSpec::Conv { out_channels, .. } => *out_channels,
                BlockSpec::Uib { out_channels, .. } => *out_channels,
            })
            .unwrap_or(self.stem_channels)
    }
}

/// MobileNetV4 image classifier.
#[derive(Clone, Debug, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
pub struct MobileNetV4 {
    pub stem: ConvNormAct,
    pub blocks: ModuleList<MobileNetBlock>,
    pub head: ConvNormAct,
    pub pool: GlobalAvgPool,
    pub classifier: Linear,
}

impl MobileNetV4 {
    pub fn new(config: &MobileNetV4Config) -> Result<Self> {
        let activation = config.activation;
        let stem = ConvNormAct::new(3, config.stem_channels, [3, 3], [2, 2], 1, activation);
        let mut channels = config.stem_channels;
        let mut blocks = Vec::with_capacity(config.blocks.len());
        for (index, spec) in config.blocks.iter().enumerate() {
            match *spec {
                BlockSpec::Conv {
                    out_channels,
                    kernel,
                    stride,
                } => {
                    blocks.push(MobileNetBlock::Conv(ConvNormAct::new(
                        channels,
                        out_channels,
                        [kernel, kernel],
                        [stride, stride],
                        1,
                        activation,
                    )));
                    channels = out_channels;
                }
                BlockSpec::Uib {
                    out_channels,
                    expand_ratio,
                    kernel,
                    stride,
                    start_depthwise,
                    se,
                } => {
                    let hidden = channels * expand_ratio.max(1);
                    blocks.push(MobileNetBlock::Uib(UibBlock::new(UibSpec {
                        in_channels: channels,
                        out_channels,
                        expand_channels: hidden,
                        kernel: [kernel, kernel],
                        stride: [stride, stride],
                        use_start_depthwise: start_depthwise,
                        use_expand: hidden != channels,
                        use_se: se,
                        activation,
                    })));
                    let _ = index;
                    channels = out_channels;
                }
            }
        }
        let head = ConvNormAct::new(
            channels,
            config.head_channels,
            [1, 1],
            [1, 1],
            1,
            activation,
        );
        let classifier =
            Linear::with_seed(config.head_channels, config.num_classes, true, 0x00C0_FFEE);
        Ok(MobileNetV4 {
            stem,
            blocks: ModuleList::from_vec(blocks),
            head,
            pool: GlobalAvgPool,
            classifier,
        })
    }

    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    pub fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        let mut hidden = self.stem.forward(x);
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        let hidden = self.head.forward(hidden);
        let pooled = self.pool.forward(hidden);
        self.classifier.forward(pooled)
    }

    /// Number of trainable parameters in the model tree.
    pub fn parameter_count(&self) -> Result<usize> {
        let mut registry = ParameterRegistry::new();
        self.visit_parameters(&ModulePath::root(), &mut registry)?;
        Ok(registry.specs().iter().map(|spec| spec.numel()).sum())
    }
}
