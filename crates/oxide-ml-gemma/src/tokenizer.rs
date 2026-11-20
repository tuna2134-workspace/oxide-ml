//! HuggingFace `tokenizer.json` support for Gemma 4.
//!
//! The model runtime consumes token ids; this wrapper makes the CLI usable with
//! plain text (`--prompt`) while keeping the reference comparison on ids.

use oxide_ml_core::{Error, Result};
use tokenizers::Tokenizer;

pub struct GemmaTokenizer {
    inner: Tokenizer,
}

impl GemmaTokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = Tokenizer::from_file(path).map_err(|error| {
            Error::Serialization(format!("cannot load tokenizer {}: {error}", path.display()))
        })?;
        Ok(GemmaTokenizer { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|error| Error::Serialization(format!("tokenizer encode: {error}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|error| Error::Serialization(format!("tokenizer decode: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORD_LEVEL: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"hello": 0, "world": 1},
        "unk_token": "<unk>"
      }
    }"#;

    #[test]
    fn encodes_and_decodes_word_level() {
        let path =
            std::env::temp_dir().join(format!("oxide-ml-tokenizer-{}.json", std::process::id()));
        std::fs::write(&path, WORD_LEVEL).unwrap();
        let tokenizer = GemmaTokenizer::from_file(&path).unwrap();
        let ids = tokenizer.encode("hello world").unwrap();
        assert_eq!(ids, vec![0, 1]);
        let text = tokenizer.decode(&ids).unwrap();
        assert!(text.contains("hello"), "{text}");
        let _ = std::fs::remove_file(path);
    }
}
