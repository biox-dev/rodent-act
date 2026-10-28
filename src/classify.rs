//! XGBoost inference over `WindowFeatures`.
//!
//! Loads the JSON model produced by `Booster.save_model("clf.json")` (the
//! standard, version-stable XGBoost JSON format) and evaluates it here - no
//! `libxgboost` dependency, so a single static binary can ship the classifier.
//!
//! Supported: `gbtree` boosters with logistic objectives, missing-value
//! handling (default left/right child).
//!
//! The window-level features come from `features::compute_window`; the feature
//! order is fixed by `features::FEATURE_NAMES` and must match the training
//! order used in `analytics/train_xgb.py`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::features::FEATURE_NAMES;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ActClass {
    /// Immobile (TST "immobility" score).
    Immobile,
    /// Mobile / climbing / struggling.
    Mobile,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Prediction {
    /// Probability of the *immobile* class (label 1 during training).
    pub immobile_prob: f32,
    pub label: ActClass,
}

impl Prediction {
    /// Re-label the prediction with a caller-supplied decision threshold.
    ///
    /// The model always produces a probability; the 0.5 cutoff is just the
    /// default. Keeping the threshold out of the model lets the UI explore
    /// stricter / looser criteria (`--threshold`) without re-scoring.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.label = if self.immobile_prob > threshold {
            ActClass::Immobile
        } else {
            ActClass::Mobile
        };
        self
    }
}

pub trait Classifier: Send {
    fn predict(&self, values: &[f32]) -> Result<Prediction>;
}

// --------------------------------------------------------------------------- //
// XGBoost JSON model
// --------------------------------------------------------------------------- //

#[derive(Deserialize)]
struct Model {
    learner: Learner,
}

#[derive(Deserialize)]
struct Learner {
    gradient_booster: Booster,
    learner_model_param: LearnerParams,
    objective: Objective,
}

#[derive(Deserialize)]
struct LearnerParams {
    base_score: String,
    #[serde(default)]
    num_feature: String,
    #[serde(default)]
    num_class: String,
}

#[derive(Deserialize)]
struct Objective {
    name: String,
}

#[derive(Deserialize)]
struct Booster {
    model: TreeModel,
}

#[derive(Deserialize)]
struct TreeModel {
    trees: Vec<Tree>,
}

#[derive(Deserialize)]
struct Tree {
    #[serde(default)]
    left_children: Vec<i32>,
    #[serde(default)]
    right_children: Vec<i32>,
    #[serde(default)]
    split_indices: Vec<i32>,
    #[serde(default)]
    split_conditions: Vec<f32>,
    /// Newer XGBoost keeps leaf weights here instead of in `split_conditions`.
    #[serde(default)]
    base_weights: Vec<f32>,
    #[serde(default)]
    default_left: Vec<i32>,
}

struct CompiledTree {
    left: Vec<i32>,
    right: Vec<i32>,
    feature: Vec<i32>,
    threshold: Vec<f32>,
    default_left: Vec<i32>,
    leaf: Vec<Option<f32>>,
}

pub struct XgbModel {
    trees: Vec<CompiledTree>,
    /// Base score in *margin* (log-odds) space.
    base_margin: f32,
    num_features: usize,
}

impl XgbModel {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading XGBoost model {}", path.display()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let m: Model = serde_json::from_str(text).context("parsing XGBoost JSON")?;
        let obj = m.learner.objective.name.to_ascii_lowercase();
        if !(obj.contains("binary:logistic") || obj.contains("reg:logistic")) {
            bail!(
                "unsupported objective `{}` (only logistic classifiers are supported)",
                m.learner.objective.name
            );
        }
        let num_class: usize = m.learner.learner_model_param.num_class.parse().unwrap_or(1);
        if num_class > 1 {
            bail!("multi-class models are not supported (num_class={num_class})");
        }

        // `base_score` may be a bare number or a bracketed vector, e.g. `[5E-1]`.
        let raw = m.learner.learner_model_param.base_score.trim().to_string();
        let cleaned = raw.trim_start_matches('[').trim_end_matches(']');
        let base_raw: f32 = cleaned
            .split(',')
            .next()
            .unwrap_or("0.5")
            .trim()
            .parse()
            .unwrap_or(0.5);
        // XGBoost 2.x stores the probability; 1.x stored log-odds. Convert
        // anything that looks like a probability into margin space.
        let base_margin = if (0.01..0.99).contains(&base_raw) {
            (base_raw / (1.0 - base_raw)).ln()
        } else {
            base_raw
        };

        let mut trees = Vec::with_capacity(m.learner.gradient_booster.model.trees.len());
        for t in m.learner.gradient_booster.model.trees {
            if t.left_children.is_empty() {
                continue;
            }
            let n = t.left_children.len();
            let mut leaf = vec![None; n];
            for i in 0..n {
                if t.left_children[i] == -1 && t.right_children[i] == -1 {
                    // Leaf value: prefer `base_weights`, fall back to
                    // `split_conditions` (the layout used by older exports).
                    let v = if t.base_weights.len() == n {
                        t.base_weights[i]
                    } else {
                        *t.split_conditions.get(i).unwrap_or(&0.0)
                    };
                    leaf[i] = Some(v);
                }
            }
            trees.push(CompiledTree {
                left: t.left_children.clone(),
                right: t.right_children.clone(),
                feature: t.split_indices,
                threshold: t.split_conditions,
                default_left: if t.default_left.is_empty() {
                    vec![0; n]
                } else {
                    t.default_left
                },
                leaf,
            });
        }
        if trees.is_empty() {
            bail!("model contains no usable trees");
        }

        let num_features = m
            .learner
            .learner_model_param
            .num_feature
            .parse::<usize>()
            .unwrap_or(FEATURE_NAMES.len());

        Ok(Self {
            trees,
            base_margin,
            num_features,
        })
    }

    /// Raw score (log-odds) before the sigmoid.
    pub fn margin(&self, values: &[f32]) -> f32 {
        let mut margin = self.base_margin;
        for t in &self.trees {
            let mut node = 0usize;
            loop {
                if let Some(v) = t.leaf[node] {
                    margin += v;
                    break;
                }
                let f = match t.feature.get(node) {
                    Some(f) if *f >= 0 => *f as usize,
                    _ => break,
                };
                let x = values.get(f).copied().unwrap_or(f32::NAN);
                let go_left = if x.is_nan() {
                    t.default_left.get(node).copied().unwrap_or(0) != 0
                } else {
                    x < t.threshold[node]
                };
                let next = if go_left { t.left[node] } else { t.right[node] };
                if next < 0 {
                    break;
                }
                node = next as usize;
            }
        }
        margin
    }
}

impl Classifier for XgbModel {
    fn predict(&self, values: &[f32]) -> Result<Prediction> {
        if values.len() < self.num_features.min(FEATURE_NAMES.len()) {
            bail!(
                "feature vector too short: got {}, model expects >= {}",
                values.len(),
                self.num_features.min(FEATURE_NAMES.len())
            );
        }
        let z = self.margin(values);
        let p = 1.0 / (1.0 + (-z).exp()); // sigmoid
        Ok(Prediction {
            immobile_prob: p,
            label: if p > 0.5 { ActClass::Immobile } else { ActClass::Mobile },
        })
    }
}

// --------------------------------------------------------------------------- //
// baseline classifier (no model file needed)
// --------------------------------------------------------------------------- //

/// Interpretable fallback used until a trained model is available:
/// mean speed below `speed_threshold` (body lengths / s) => immobile.
pub struct ThresholdClassifier {
    pub speed_index: usize,
    pub speed_threshold: f32,
}

impl Default for ThresholdClassifier {
    fn default() -> Self {
        let idx = FEATURE_NAMES
            .iter()
            .position(|n| *n == "speed_mean")
            .unwrap_or(0);
        Self {
            speed_index: idx,
            speed_threshold: 0.30,
        }
    }
}

impl Classifier for ThresholdClassifier {
    fn predict(&self, values: &[f32]) -> Result<Prediction> {
        let s = *values
            .get(self.speed_index)
            .ok_or_else(|| anyhow::anyhow!("feature vector too short"))?;
        // logistic ramp around the threshold; 0.1 body-length/s wide
        let p = 1.0 / (1.0 + ((s - self.speed_threshold) / 0.1).exp());
        Ok(Prediction {
            immobile_prob: p,
            label: if p > 0.5 { ActClass::Immobile } else { ActClass::Mobile },
        })
    }
}

/// Load a model, falling back to the baseline when no file is given.
pub fn load_or_baseline(path: Option<&Path>) -> Result<Box<dyn Classifier>> {
    match path {
        Some(p) if p.exists() => Ok(Box::new(XgbModel::load(p)?)),
        Some(p) => bail!("classifier model not found: {}", p.display()),
        None => Ok(Box::new(ThresholdClassifier::default())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny 1-tree model: splits on feature 0 (`speed_mean`, the first
    /// entry of `FEATURE_NAMES`) at 0.5 with leaves -2 / +2.
    const MODEL: &str = r#"{
      "learner": {
        "learner_model_param": {"base_score": "5E-1", "num_feature": "18", "num_class": "1"},
        "objective": {"name": "binary:logistic"},
        "gradient_booster": {
          "model": {
            "trees": [
              {
                "left_children": [1, -1, -1],
                "right_children": [2, -1, -1],
                "split_indices": [0, 0, 0],
                "split_conditions": [0.5, -2.0, 2.0],
                "default_left": [1, 0, 0]
              }
            ]
          }
        }
      }
    }"#;

    fn features_with_speed(speed: f32) -> Vec<f32> {
        let mut v = vec![0.0; FEATURE_NAMES.len()];
        let idx = FEATURE_NAMES.iter().position(|n| *n == "speed_mean").unwrap();
        v[idx] = speed;
        v
    }

    #[test]
    fn evaluates_split() {
        let m = XgbModel::from_json(MODEL).unwrap();
        let slow = m.predict(&features_with_speed(0.1)).unwrap();
        let fast = m.predict(&features_with_speed(0.9)).unwrap();
        // slower speed -> left leaf (-2) -> mobile-side probability, because
        // this synthetic tree is oriented that way; the point of the test is
        // that the split is evaluated, not the semantics.
        assert!((slow.immobile_prob - 0.119).abs() < 0.01, "{slow:?}");
        assert!((fast.immobile_prob - 0.881).abs() < 0.01, "{fast:?}");
        assert_ne!(slow.label, fast.label);
    }

    #[test]
    fn rejects_unsupported_objective() {
        let bad = MODEL.replace("binary:logistic", "multi:softmax");
        assert!(XgbModel::from_json(&bad).is_err());
    }

    #[test]
    fn baseline_threshold() {
        let c = ThresholdClassifier::default();
        assert_eq!(
            c.predict(&features_with_speed(0.05)).unwrap().label,
            ActClass::Immobile
        );
        assert_eq!(
            c.predict(&features_with_speed(1.50)).unwrap().label,
            ActClass::Mobile
        );
    }
}
