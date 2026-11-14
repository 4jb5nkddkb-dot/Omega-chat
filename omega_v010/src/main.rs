//! omega-world-bridge — Rust side of the Omega v0.1.0 closed loop.
//!
//! The kernel (omega-cognitive-kernel, crates.io 0.5.6) runs the
//! L0/L1/L2 cognition. This binary injects the real world into it:
//!   * FileWorldObserver — reads observation.json written by the Python
//!     sidecar (world summary + model reply + grounded error score).
//!   * FileWorldExecutor — writes action.json (chosen action + params)
//!     for the sidecar to execute in the world.
//! After each kernel step, the binary waits for the sidecar's
//! outcome.json (grounded error for the executed action) and feeds
//! it back as a strategy reward via `record_reward` — the grounding
//! channel. L1 self-modification stays OFF (omega.toml); the loop
//! still learns through strategy weights + case memory + L2 nudges,
//! and everything is audited (audit.jsonl / trajectory.jsonl).
//!
//! Experiment A (workbench only): the kernel only ever picks stabilize,
//! because a stabilize success keeps raising its weight (x1.05 per
//! success) while explore is never registered, so
//! `set_strategy_weight("explore", _)` silently returns false.
//! Optional flags pin the levers after each step:
//!   --stab-weight <f64>     pin stabilize's weight (registered, works)
//!   --explore-bonus <f64>   pin RuleEffect.explore_bonus (clamped
//!                           [-1.5, 1.5]; the lever that can actually
//!                           move explore, since its weight is unpinned)
//!   --explore-weight <f64>  one pre-loop set_strategy_weight("explore")
//!                           attempt, logged (expected false until
//!                           explore has been taken once).
//! Without the flags the loop behaves exactly as before.

use omega_cognitive_kernel::{
    Kernel, KernelConfig, OmegaError, Result,
    l0::executor::Executor,
    l0::observer::Observer,
    types::{Action, Modality, Observation},
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

const OBSERVE_TIMEOUT_SECS: u64 = 60;
const OUTCOME_TIMEOUT_SECS: u64 = 300;

struct FileWorldObserver {
    dir: PathBuf,
    n: u64,
}

impl Observer for FileWorldObserver {
    fn observe(&mut self) -> Result<Vec<Observation>> {
        self.n += 1;
        let path = self.dir.join("observation.json");
        let content = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or_else(|| json!({"status": "waiting", "step": self.n}));
        Ok(vec![Observation {
            timestamp: chrono::Utc::now(),
            modality: Modality::Structured,
            content,
            confidence: 0.9,
            source: "world-bridge".into(),
        }])
    }
}

struct FileWorldExecutor {
    dir: PathBuf,
    n: u64,
}

impl Executor for FileWorldExecutor {
    fn execute(&mut self, actions: &[Action]) -> Result<Vec<serde_json::Value>> {
        let mut out = Vec::new();
        for a in actions {
            if !a.constraints_checked {
                return Err(OmegaError::ExecutionBlocked(format!(
                    "{} (constraints not checked)",
                    a.name
                )));
            }
            self.n += 1;
            let rec = json!({
                "step": self.n,
                "action": a.name,
                "params": a.params,
            });
            let text = serde_json::to_string_pretty(&rec)
                .map_err(|e| OmegaError::ExecutionBlocked(format!("serialize action: {e}")))?;
            fs::write(self.dir.join("action.json"), text)
                .map_err(|e| OmegaError::ExecutionBlocked(format!("write action.json: {e}")))?;
            out.push(json!({"status": "dispatched", "action": a.name, "step": self.n}));
        }
        Ok(out)
    }
}

fn wait_for_outcome(dir: &PathBuf, step: u64) -> Option<serde_json::Value> {
    let path = dir.join("outcome.json");
    let start = Instant::now();
    loop {
        if let Ok(s) = fs::read_to_string(&path) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                if v.get("step").and_then(|x| x.as_u64()) == Some(step) {
                    return Some(v);
                }
            }
        }
        if start.elapsed() > Duration::from_secs(OUTCOME_TIMEOUT_SECS) {
            return None;
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut dir = PathBuf::from("run");
    let mut steps: u32 = 8;
    let mut goal = "act coherently in the little world as Omega".to_string();
    let mut config: Option<PathBuf> = None;
    let mut state: Option<PathBuf> = None;
    let mut stab_weight: Option<f64> = None;
    let mut explore_bonus: Option<f64> = None;
    let mut explore_weight: Option<f64> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--steps" => {
                steps = args[i + 1].parse().unwrap_or(8);
                i += 2;
            }
            "--goal" => {
                goal = args[i + 1].clone();
                i += 2;
            }
            "--config" => {
                config = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--state" => {
                state = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--stab-weight" => {
                stab_weight = args[i + 1].parse().ok();
                i += 2;
            }
            "--explore-bonus" => {
                explore_bonus = args[i + 1].parse().ok();
                i += 2;
            }
            "--explore-weight" => {
                explore_weight = args[i + 1].parse().ok();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    fs::create_dir_all(&dir)
        .map_err(|e| OmegaError::ExecutionBlocked(format!("mkdir run dir: {e}")))?;
    let state_path = state.unwrap_or_else(|| dir.join("state.json"));

    let obs_path = dir.join("observation.json");
    let start = Instant::now();
    while !obs_path.exists() {
        if start.elapsed() > Duration::from_secs(OBSERVE_TIMEOUT_SECS) {
            eprintln!("waiting for observation.json timed out (sidecar not started?)");
            std::process::exit(2);
        }
        thread::sleep(Duration::from_millis(200));
    }

    let observer = FileWorldObserver { dir: dir.clone(), n: 0 };
    let executor = FileWorldExecutor { dir: dir.clone(), n: 0 };
    let mut kernel = Kernel::with_adapters_and_path(observer, executor, &state_path);
    if let Some(cfg_path) = &config {
        let cfg = KernelConfig::load(cfg_path)?;
        kernel.apply_config(&cfg);
    }
    println!("goal   : {goal}");
    println!("steps  : {steps}");
    println!("state  : {}", state_path.display());
    println!("self_modify: {}", kernel.inner.l1.self_modify_enabled);
    if let Some(ew) = explore_weight {
        // Diagnostic: explore is unregistered until taken once, so this
        // is expected to return false on a fresh state. Logged, not fatal.
        let ok = kernel.set_strategy_weight("explore", ew);
        println!("pin    : pre-loop set explore weight {ew} -> registered={ok}");
    }
    if let Some(b) = explore_bonus {
        let b = b.clamp(-1.5, 1.5);
        kernel.inner.l0.memory.cognitive_effect.explore_bonus = b;
        println!("pin    : explore_bonus={b}");
    }
    if let Some(sw) = stab_weight {
        println!("pin    : stabilize weight={sw} (applied after each step)");
    }

    for step in 1..=steps {
        let r = kernel.step(&goal)?;
        let action = r
            .trajectory
            .actions
            .first()
            .map(|a| a.name.clone())
            .unwrap_or_else(|| "-".into());
        let outcome = wait_for_outcome(&dir, step as u64);
        let (gerr, reply) = match &outcome {
            Some(v) => (
                v.get("grounded_error").and_then(|x| x.as_f64()).unwrap_or(1.0),
                v.get("reply")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string(),
            ),
            None => (1.0, String::new()),
        };
        let reward = (1.0 - gerr).clamp(0.0, 1.0);
        kernel.inner.l0.memory.record_reward(&action, reward);
        // Experiment A pins (after each step, so the kernel's own
        // absorb/bump learning cannot drift them back mid-run).
        let mut pin_note = String::new();
        if let Some(sw) = stab_weight {
            let ok = kernel.set_strategy_weight("stabilize", sw);
            pin_note.push_str(&format!(" pin_stab={ok}"));
        }
        if let Some(b) = explore_bonus {
            kernel.inner.l0.memory.cognitive_effect.explore_bonus = b.clamp(-1.5, 1.5);
            pin_note.push_str(&format!(" pin_bonus={}", b.clamp(-1.5, 1.5)));
        }
        let _ = fs::remove_file(dir.join("outcome.json"));
        println!(
            "step {step:>2} | act={action:<9} | grounded_err={gerr:.2} reward={reward:.2} | avg_err={:.3} damp={:.2} patched={}{pin_note} | reply={reply}",
            kernel.inner.l0.memory.cas.avg_prediction_error,
            kernel.inner.l0.memory.cognitive_effect.error_damping,
            r.l1.patched,
        );
        if outcome.is_none() {
            eprintln!("step {step}: outcome.json timed out; continuing with reward 0");
        }
    }
    kernel.save()?;
    let st = kernel.status();
    println!("\n-- status --");
    println!(
        "steps={} avg_err={:.3} damp={:.3} cases={} strategies={} ckpts={}",
        st.total_steps,
        st.avg_prediction_error,
        st.error_damping,
        st.case_count,
        st.strategy_count,
        st.checkpoint_count
    );
    println!(
        "L2: in_fp={} dwell={} align={:.3} self_modify={}",
        st.in_fixed_point, st.dwell_steps, st.last_alignment_score, st.self_modify_enabled
    );
    Ok(())
}
