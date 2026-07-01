"""Two-page project report: page 1 = concise 3-phase bullets, page 2 = 3 figures."""
import os
import matplotlib
from fpdf import FPDF

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
REG = os.path.join(FONT_DIR, "DejaVuSans.ttf")
BLD = os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")
ITA = os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")
FIGDIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(FIGDIR, "PROJECT_REPORT.pdf")

INK = (33, 37, 41)
BODY = (48, 52, 56)
GREY = (110, 116, 122)
C1 = (37, 99, 175)
C2 = (26, 140, 88)
C3 = (200, 110, 25)
CB = (90, 60, 140)

INTRO = ("We trained a small neural network, by reinforcement learning (RL), to keep a pendulum "
         "upright while its base moves — built up over three projects, each reusing the last.")

PHASES = [
    (C1, "Phase 1 · Flat ground — a robust balancer, deployed on hardware", [
        "**Problem** — a classical controller (LQR) balanced in simulation but failed on the real "
        "motor: friction winds the arm into its cable limit.",
        "**Algorithm** — reinforcement learning with TQC (Truncated Quantile Critics), an off-policy "
        "actor-critic (SAC family) whose critic predicts a whole distribution of future rewards and "
        "drops the most optimistic ones to stay honest; exploration uses gSDE (state-dependent noise) "
        "for smoother, more transferable behaviour.",
        "**Smooth control** — CAPS: a reward penalty on abrupt command changes (−0.02·(Δa)²) so the "
        "motor commands stay smooth and transfer to a real motor (jerky policies buzz/vibrate on "
        "hardware).",
        "**Deploy** — we run the trained policy (a tiny neural network, or MLP) directly on the "
        "microcontroller (ESP32), with no PC in the loop; removing that communication latency is what "
        "makes real-time balancing reliable.",
        "**Result** — full swing-up and balance on the real device.",
    ]),
    (C2, "Phase 2 · One tilting axis — add a sensor, keep the skill  ·  verified 91.5%", [
        "**Challenge** — the base tilts ~±20° on one axis; the pole must stay upright to true "
        "gravity, not to the tilting base.",
        "**Sensor** — added a BNO086 IMU to read board tilt, and retargeted the reward to true "
        "gravity-vertical.",
        "**Reuse, don't restart** — instead of training a new controller from scratch for the harder "
        "task, we start from the already-working Phase-1 controller and fine-tune it (a 'warm "
        "start') — this saves training effort and keeps its proven skills.",
        "**Retention** — fine-tuning risks forgetting those skills, so during training we replay the "
        "old controller's successful experiences and gently pull the new actions toward the old ones. "
        "The critic learns faster than the actor — it must relearn the new task quickly, while the "
        "actor changes slowly so it doesn't wreck the proven policy. (gSDE off here — it caused an "
        "'entropy collapse'.)",
        "**Verify** — 500 fresh trials per condition; success = stays balanced ≥80% of the final 2 s "
        "(not a brief catch); reported with Wilson 95% confidence intervals (an error-bar on each "
        "rate that accounts for the finite number of trials).",
        "**Result** — 91.5% sustained success through one-axis tilt.",
    ]),
    (C3, "Phase 3 · Two tilting axes — diagnose and fix a hidden failure  ·  current", [
        "**Challenge** — the board now tilts in both roll and pitch, fast; capped at ±15° and "
        "~120°/s to match the real moving board.",
        "**Failure** — the critic (a 'coach' that scores each situation) was out of its depth on the "
        "new motion, and because it scores itself (bootstrapping) the errors snowballed — rating "
        "clearly-good states as bad — so the controller got useless guidance and stalled.",
        "**Fix** — throw away the broken coach and start it fresh, then re-train it the easy way with "
        "the controller held still. With the controller fixed, the coach faces one stable question: "
        "from a given situation, how much reward does this fixed controller actually collect? We run "
        "the frozen controller in simulation and train the coach to match those real outcomes — a "
        "grounded task, free of the self-referential feedback that caused the blow-up — and only once "
        "its scores match reality do we let the controller improve again. We also shorten the "
        "planning horizon (a lower discount factor) for steadier scores and clip gradient sizes to "
        "prevent rare blow-ups.",
        "**Robustness** — we train several independent runs and keep the best; most reach the same "
        "strong envelope (Figure 3).",
        "**Result** — balances through fast two-axis board motion (see figures).",
    ]),
]

CLOSING = ("**What made it work:**  build on the previous result instead of restarting  ·  find the "
           "true cause cheaply before spending compute  ·  trust only verified numbers.")

FIGS = [
    ("figure_10V_trace.png",
     "Figure 1. One episode: the board swings in roll and pitch (orange) while the pole stays near "
     "vertical (blue) — the task being solved."),
    ("figure_10V_envelope.png",
     "Figure 2. Capability (best controller): sustained success stays high (96–99%) across the whole "
     "speed range, up to fast two-axis motion (500-trial verification; shaded = 95% confidence)."),
    ("figure_10V_seeds.png",
     "Figure 3. Robustness: five independently trained controllers are tightly matched up to ~80°/s "
     "and diverge only at the very fastest speed — the strong result is the method's, not luck."),
]

L, R = 18, 192
W = R - L


class PDF(FPDF):
    def footer(self):
        self.set_y(-10)
        self.set_font("D", "", 7)
        self.set_text_color(*GREY)
        self.cell(0, 6, "Balancing a pendulum on a moving board — RL methods & results", align="C")
        self.cell(0, 6, f"{self.page_no()}", align="R")


pdf = PDF(format="A4")
pdf.set_auto_page_break(True, margin=12)
pdf.add_font("D", "", REG)
pdf.add_font("D", "B", BLD)
pdf.add_font("D", "I", ITA)

pdf.add_page()
pdf.set_xy(L, 13)
pdf.set_font("D", "B", 15.5)
pdf.set_text_color(*INK)
pdf.cell(0, 7, "Learning to Balance a Pendulum on a Moving Board")
pdf.ln(7)
pdf.set_x(L)
pdf.set_font("D", "I", 8.8)
pdf.set_text_color(*GREY)
pdf.cell(0, 4.5, "Methods and results, and why each method mattered")
pdf.ln(5.2)
pdf.set_draw_color(210, 213, 216)
pdf.set_line_width(0.3)
pdf.line(L, pdf.get_y(), R, pdf.get_y())
pdf.ln(3.4)

pdf.set_x(L)
pdf.set_font("D", "", 8.6)
pdf.set_text_color(*BODY)
pdf.multi_cell(W, 4.3, INTRO)
pdf.ln(3)


def bullet(text):
    y0 = pdf.get_y()
    pdf.set_font("D", "", 8.6)
    pdf.set_text_color(*BODY)
    pdf.set_xy(L + 1.5, y0)
    pdf.cell(3.2, 4.3, "•")
    pdf.set_xy(L + 4.7, y0)
    pdf.multi_cell(W - 4.7, 4.3, text, markdown=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.7)


for color, heading, bullets in PHASES:
    pdf.set_x(L)
    pdf.set_font("D", "B", 10)
    pdf.set_text_color(*color)
    pdf.multi_cell(W, 5.2, heading)
    pdf.ln(1.1)
    for b in bullets:
        bullet(b)
    pdf.ln(1.8)

pdf.ln(0.3)
pdf.set_draw_color(210, 213, 216)
pdf.line(L, pdf.get_y(), R, pdf.get_y())
pdf.ln(2.6)
pdf.set_x(L)
pdf.set_font("D", "", 8.6)
pdf.set_text_color(*CB)
pdf.multi_cell(W, 4.3, CLOSING, markdown=True)

# ---- page 2 ----
if all(os.path.exists(os.path.join(FIGDIR, f)) for f, _ in FIGS):
    pdf.add_page()
    pdf.set_xy(L, 13)
    pdf.set_font("D", "B", 13)
    pdf.set_text_color(*INK)
    pdf.cell(0, 7, "Results")
    pdf.ln(7)
    for fname, cap in FIGS:
        iw = 132
        pdf.image(os.path.join(FIGDIR, fname), x=(210 - iw) / 2, w=iw)
        pdf.ln(0.8)
        pdf.set_x(L)
        pdf.set_font("D", "I", 7.8)
        pdf.set_text_color(70, 74, 78)
        pdf.multi_cell(W, 3.7, cap, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3.2)
    print("page 2 (figures) INCLUDED")
else:
    print("page 2 SKIPPED — missing:",
          [f for f, _ in FIGS if not os.path.exists(os.path.join(FIGDIR, f))])

pdf.output(OUT)
print("wrote", OUT)
