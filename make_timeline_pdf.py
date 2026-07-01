"""Generate a concise timeline PDF of the Furuta RL project lineage."""
import os
import matplotlib
from fpdf import FPDF

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
REG = os.path.join(FONT_DIR, "DejaVuSans.ttf")
BLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
ITA = os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PROJECT_TIMELINE.pdf")

# (date, title, description)
P1 = [
    ("Goal", "Level-ground Furuta on GM3506 gimbal",
     "Classical LQR balances in sim but FAILS on hardware — pivot friction winds the arm to the "
     "±180° cable limit (no integral action vs the friction offset)."),
    ("Method", "One TQC policy for swing-up AND balance",
     "MuJoCo + wide domain randomization + curriculum. Obs 6-D [cosθ, sinθ, θ̇, φ, φ̇, prev_action]; "
     "reward = cos(θ) backbone + CAPS action-smoothness; bonus/success gated on arm < 90° (kills the "
     "'balance at the cable edge' loophole)."),
    ("Findings", "Realism fears were mostly pessimistic",
     "Actuator step-test: real lag ~15 ms, already modeled → swing-up transfers. Ablation flagged "
     "'motor authority matters' (KM×0.75 → 17% success) — a precursor to the 2D 11 V result."),
    ("2026-06-24", "RESULT: deployed on-chip, works on hardware",
     "Sim-trained TQC runs standalone on the ESP32 (MODE_RL) doing the full task; ~88% under "
     "randomized DR. This is the 'trained from nothing' origin."),
]

P2 = [
    ("Goal", "Board tilts ±20–30° about ONE axis",
     "Keep balancing to TRUE gravity-vertical through the motion. Add a BNO086 IMU (board tilt "
     "β, β̇) → 8-D obs; retarget reward from base-frame 'up' to true vertical."),
    ("Phase 0", "Sim feasibility confirmed",
     "±30° is achievable but orientation-dependent (arm at φ≈90° is the hard case where tilt most "
     "drives the pole)."),
    ("2026-06-26", "Challenge: entropy collapse",
     "Stage-0 learned then oscillated/dropped — ent_coef ran away under gSDE + auto target-entropy. "
     "Fix: turn gSDE off. Also learned: use nenv=8, not 16 (better sample efficiency here)."),
    ("06-27→29", "Phase C: metric fix + retention-aware fine-tuning",
     "Corrected success = balanced ≥80% of the final 2 s (not just a brief catch). Framed fine-tuning "
     "as a forgetting problem → teacher replay + behavior-cloning (RetentionTQC)."),
    ("2026-06-29", "Verified master: 91.5%",
     "Clean ±20° free-arm master (clean20_master_verified91p5) — 915/1000 sustained. The anchor model "
     "for all later work."),
    ("2026-06-29", "Key finding: action DELAY is the DR bottleneck",
     "Ablation: mechanical/sensor DR barely matters; delay does (2-step → ~40%, 3-step → ~0%). It's "
     "non-Markov — a bounded residual can't fix an unobserved phase lag. Residual RL preserved but "
     "did NOT beat the clean master."),
]

P3 = [
    ("Setup", "Two-axis (roll + pitch) board, 10-D obs",
     "Nested-gimbal model + continuous motion generator + BNO086 model. Training-free 1D→2D "
     "warm-start (transferred master, verified equivalent to 2e-6)."),
    ("Step 4", "Warm-start: great everywhere except fast pitch",
     "~99% on level / roll / slow / static, but ~10% on FAST continuous pitch. That single gap "
     "became the whole problem to solve."),
    ("Step 5", "Diagnosis: it's the CRITIC, not capability",
     "Probe showed fast pitch is LEARNABLE (torque headroom at failure). Real cause: the transferred "
     "critic diverges NEGATIVE (predicts −1800 for states worth +1000), starving the actor."),
    ("Phase A/B", "Fix: reset the critic + warmup + γ=0.99",
     "Discard the transferred critic, re-initialize it, warm it up with the actor frozen until "
     "calibrated, lower γ to 0.99. Divergence gone; co-training clears the ±10° regimes."),
    ("Step 6", "Envelope decision: ±10°, up to 120°/s",
     "±15°-fast is a REAL physics wall at 6 V (servo test confirmed it's not a sim artifact) → "
     "scoped the deployment target to ±10°."),
    ("Scale", "5 seeds parallel on the server",
     "5× RTX 6000 Ada. Verified local seed 0 (500 ep): ~100% slow/mid, ~34% at both ±10° 120°/s, "
     "with a calibrated critic. En route: 4/5 seeds NaN-crashed → added gradient clipping."),
    ("2026-06-30", "11 V discovery: authority erases the wall",
     "Motor runs ~11 V (sim used a conservative 6 V ≈ 1.83× less torque). Retrained at 11 V "
     "(teacher-free) → cleared the ENTIRE ±15° ladder at 0.93–0.97 (vs 0.10–0.33 at 6 V). Authority, "
     "not physics, was the limit. Pending 500-ep verify + a hardware thermal check."),
]

METHODS = [
    ("Transfer, don't restart",
     "Warm-start each project from the previous project's VERIFIED model instead of training from "
     "scratch — preserves proven swing-up/balance and avoids TQC seed-variance failures."),
    ("TQC + CAPS + true-vertical reward",
     "Truncated Quantile Critics (off-policy, tiny deployable actor); CAPS action-smoothness so "
     "control transfers to a real motor; reward on TRUE gravity-vertical once the base tilts."),
    ("Critic reset + frozen-actor warmup + γ=0.99",
     "The key 2D fix. Discard the out-of-distribution transferred critic, re-initialize it, and "
     "calibrate it with the actor FROZEN before it drives the actor; lower γ for stable, "
     "lower-variance targets. Cures the negative-divergence stall."),
    ("Retention fine-tuning (teacher + behavior-cloning)",
     "Fine-tune without forgetting: mix successful old-policy transitions into replay and add an "
     "MSE loss pulling the actor toward them. Disable when the dynamics change (e.g. 11 V)."),
    ("Soft, advancing curriculum",
     "Ramp the disturbance in small steps; advance on a soft success threshold OR a timeout that "
     "ADVANCES (never kills) the seed — avoids the brittle-gate trap that stalled early runs."),
    ("Gradient clipping",
     "Clip the gradient norm to stop the exploding-gradient NaN divergence — essential with a "
     "re-initialized critic (it crashed 4/5 seeds before we added it)."),
    ("Verify before you trust",
     "Select on SUSTAINED success (balanced ≥80% of the final 2 s), over ≥500 fresh episodes with "
     "Wilson 95% CIs; keep best-per-stage checkpoints, never the final one. Small peaks are noise."),
    ("Cheap diagnostic probes",
     "Targeted probes — Q vs return-to-go calibration, action-saturation at failure — separate a "
     "training bug from a physics/authority limit BEFORE spending GPU hours."),
    ("Reproducible on-chip deployment",
     "Export the actor MLP to a C header, verify NumPy inference vs SB3 to <1e-6 on CPU, and "
     "rebuild the identical observation in firmware at 200 Hz. This is what put P1 on hardware."),
]

LESSONS = [
    "Transfer, don't restart — every project warm-started from the previous verified model.",
    "Diagnose with cheap probes before spending compute.",
    "Capability vs training: 'not saturated at failure' = trainable; 'authority ablation kills it' = physics/hardware.",
    "Trust 500-episode verified numbers + Wilson CIs — never a small training-eval peak.",
    "Instrument the mechanism: Q-vs-return, retention-loss magnitude, log-probs — bugs are invisible in success rate alone.",
]

# colors
INK = (33, 37, 41)
GREY = (110, 116, 122)
C1 = (37, 99, 175)    # P1 blue
C2 = (26, 140, 88)    # P2 green
C3 = (200, 110, 25)   # P3 orange
CM = (17, 122, 128)   # methods teal
CL = (90, 60, 140)    # lessons purple


class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("D", "", 7)
        self.set_text_color(*GREY)
        self.cell(0, 6, "Furuta Pendulum RL — project lineage · generated 2026-06-30",
                  align="C")
        self.cell(0, 6, f"{self.page_no()}", align="R")


pdf = PDF(format="A4")
pdf.set_auto_page_break(True, margin=16)
pdf.add_font("D", "", REG)
pdf.add_font("D", "B", BLD)
pdf.add_font("D", "I", ITA)
pdf.add_page()
L, R = 15, 195
W = R - L

# title
pdf.set_xy(L, 14)
pdf.set_font("D", "B", 19)
pdf.set_text_color(*INK)
pdf.cell(0, 9, "Furuta Pendulum RL — Project Timeline")
pdf.ln(8)
pdf.set_x(L)
pdf.set_font("D", "I", 10)
pdf.set_text_color(*GREY)
pdf.cell(0, 6, "From a level-ground balancer to two-axis tilt — methods, challenges, results")
pdf.ln(9)
pdf.set_draw_color(*GREY)
pdf.set_line_width(0.3)
pdf.line(L, pdf.get_y(), R, pdf.get_y())
pdf.ln(3)


def section(title, subtitle, color):
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.ln(2)
    y = pdf.get_y()
    pdf.set_fill_color(*color)
    pdf.rect(L, y, W, 8, style="F")
    pdf.set_xy(L + 2.5, y)
    pdf.set_font("D", "B", 11.5)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, title)
    pdf.set_font("D", "I", 9)
    pdf.set_xy(L, y)
    pdf.cell(W - 2.5, 8, subtitle, align="R")
    pdf.ln(10)


def entry(date, title, desc, color):
    # keep an entry from splitting awkwardly near the bottom
    if pdf.get_y() > 262:
        pdf.add_page()
    x0 = L + 3
    # date + title line
    pdf.set_x(x0)
    pdf.set_font("D", "B", 9.5)
    pdf.set_text_color(*color)
    dw = pdf.get_string_width(date) + 3
    pdf.cell(dw, 5.5, date)
    pdf.set_text_color(*INK)
    pdf.set_font("D", "B", 9.5)
    pdf.multi_cell(W - 3 - dw, 5.5, title, new_x="LMARGIN", new_y="NEXT")
    # description
    pdf.set_x(x0 + 2)
    pdf.set_font("D", "", 8.7)
    pdf.set_text_color(60, 64, 68)
    pdf.multi_cell(W - 5, 4.6, desc, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.6)


section("PROJECT 1  ·  Level-ground Furuta", "deployed, working on hardware", C1)
for e in P1:
    entry(*e, C1)

section("PROJECT 2  ·  One-axis tilting board (1D)", "verified 91.5% master", C2)
for e in P2:
    entry(*e, C2)

section("PROJECT 3  ·  Two-axis board (2D)", "2026-06-30 — in progress", C3)
for e in P3:
    entry(*e, C3)


def method_item(name, desc, color):
    if pdf.get_y() > 260:
        pdf.add_page()
    x0 = L + 3
    pdf.set_x(x0)
    pdf.set_font("D", "B", 9.5)
    pdf.set_text_color(*color)
    pdf.multi_cell(W - 3, 5.3, name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(x0 + 2)
    pdf.set_font("D", "", 8.7)
    pdf.set_text_color(60, 64, 68)
    pdf.multi_cell(W - 5, 4.6, desc, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.6)


section("SUCCESSFUL METHODS", "what worked, and why", CM)
for m in METHODS:
    method_item(*m, CM)

# lessons box
if pdf.get_y() > 235:
    pdf.add_page()
pdf.ln(2)
y = pdf.get_y()
pdf.set_fill_color(*CL)
pdf.rect(L, y, W, 8, style="F")
pdf.set_xy(L + 2.5, y)
pdf.set_font("D", "B", 11.5)
pdf.set_text_color(255, 255, 255)
pdf.cell(0, 8, "KEY LESSONS")
pdf.ln(11)
pdf.set_font("D", "", 9)
pdf.set_text_color(*INK)
for s in LESSONS:
    pdf.set_x(L + 3)
    pdf.set_text_color(*CL)
    pdf.set_font("D", "B", 9)
    pdf.cell(4, 4.8, "•")
    pdf.set_text_color(60, 64, 68)
    pdf.set_font("D", "", 9)
    pdf.multi_cell(W - 7, 4.8, s, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.8)

pdf.ln(2)
pdf.set_x(L)
pdf.set_font("D", "I", 8.5)
pdf.set_text_color(*GREY)
pdf.multi_cell(W, 4.6,
               "NOW: 11 V verification + five 6 V baseline seeds training on the server. "
               "Current best deployable model: local 2D seed 0 (verified). "
               "Full detail: PROJECT_LINEAGE_METHODS_AND_LESSONS.md.")

pdf.output(OUT)
print("wrote", OUT)
