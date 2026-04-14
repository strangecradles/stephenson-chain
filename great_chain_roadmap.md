# The Great Chain → Tendon-Driven Finger — Newton Project Roadmap

A two-week, two-phase plan: validate Newton's cable solver on freely rotating closed chains (the *Seveneves* warm-up), then apply the same primitive to a tendon-driven robotic finger — the simulation problem that next-generation dexterous humanoid hands actually depend on.

The arc is the artifact: curiosity-driven physics question → analytic validation → real robotics application → multi-solver future-work hook. Each phase strengthens the next, and together they tell a much stronger story than either project alone.

---

## Vision

The project has two phases connected by one technical primitive (Newton's cable solver) and one arc (curiosity → validation → application).

**Phase 1 — The Great Chain.** Investigate the dynamics of freely rotating closed chains in zero gravity, motivated by the Great Chain in *Seveneves*. Stephenson explicitly flagged spinning chain dynamics as under-studied in the acknowledgments, and Newton's cable solver is the first widely available tool that lets you actually study them. Validate Newton against the analytic continuous-rope prediction, characterize the discrete-to-continuous transition as a function of N, observe stability modes. This phase is the curiosity-driven warm-up that builds your fluency with the cable solver and gives you analytic baselines you can trust.

**Phase 2 — The Tendon-Driven Finger.** Once you trust the cable solver, apply it to the simulation problem that matters most for current robotics: cable-driven dexterous manipulation. Almost every serious humanoid hand (Shadow, Allegro, the hands on Figure / 1X / Apptronik / DeepMind robots) is tendon-driven — actuators at the base, force transmitted through cables routed across joints — because that's the only way to get human-like dexterity-to-mass ratios. Simulating these systems accurately and differentiably is an open problem because cables interact with their routing surfaces through frictional contact along their length, slack-to-tension transitions produce gradient pathologies, and shared tendons couple multiple joints non-trivially. Newton's cable solver is in a unique position to address this gap. Build a single tendon-driven finger and run experiments that exercise exactly these hard cases.

**Why the two phases connect.** The narrative is the trajectory of a researcher's curiosity: started with a clean physics question (does a spinning chain hold its shape), validated the simulator on it, and asked whether the same primitive could solve a much harder real-world problem (cable-driven manipulation). That arc is the same one Frey himself probably went through on the way to building the cable solver in the first place. He'll recognize it.

The artifact has three audiences stacked on top of each other:

- **The physics audience.** A clean numerical study of an under-studied classical problem with analytic baselines, *plus* a real engineering application built on the validated tool.
- **The Newton audience (Erik Frey, his team).** An honest early-adopter report on Newton's cable solver — what it does well, where it's rough, what it would need to make first-class — applied to a problem (tendon-driven manipulation) that maps directly onto their team's actual work.
- **The vision audience (everyone you eventually show this to).** A demonstration that you think about embodied physics, simulation infrastructure, and dexterous manipulation as one connected stack, and that you act on your curiosity by building real things.

The artifact you ship is two working simulations, a set of experiments across both, and a single 4–5 page technical writeup that tells the connecting story. It is **not** a polished paper. It is an honest report from an early user who built something interesting and is sharing what they found.

---

## What You Are Building

### Phase 1 — The Great Chain (days 1–5)

A parametrizable Newton scene that constructs a closed chain of N rigid links connected by flexible joints, gives it an initial angular velocity, and runs the simulation. Three focused experiments:

1. **Steady-state validation** — does the simulated chain match the continuous-rope analytic prediction T = μω²r² at large N?
2. **Discrete-to-continuous interpolation** — sweep N from 2 to ~128 and characterize the transition. This is the headline figure for Phase 1: a grid of steady-state shapes at increasing N.
3. **One stability observation** — perturb the steady-state ring at one rotation rate and report what mode appears. Don't try to do a full stability sweep; just one observation.

Phase 1 ends when you trust Newton's cable solver enough to build a real application on top of it. Don't let it sprawl.

### Phase 2 — The Tendon-Driven Finger (days 6–11)

A single tendon-driven finger built from Newton primitives: three rigid links (phalanges) connected by two pin joints, a passive return spring at each joint, and one Newton cable routed along the underside from a fixed anchor at the base, through via points or sheaths at each joint, terminating at the fingertip. Pulling the cable curls the finger; releasing it lets the springs straighten it.

This minimal scene captures the essential challenges of cable-driven actuation: force transmission through a flexible element, joint coupling through a shared tendon, contact between cable and routing surfaces, and slack-to-tension transitions.

Three focused experiments, in order of priority:

1. **Force transmission validation.** Apply a known cable tension, hold the finger in static equilibrium against an external load, measure joint angles and contact forces, compare to the textbook tendon-driven mechanism prediction. This is your validation experiment and the most likely to surface a Newton bug — which would itself be valuable.
2. **Slack-to-tension gradient pathology.** Drive the cable through a slack regime each cycle, attempt to backprop through the simulation, characterize where Newton's gradients are well-behaved and where they break. This is the experiment Frey will read most carefully — slack transitions are exactly the contact-pathology problem his team is wrestling with at scale.
3. **Underactuated grasping (headline visualization for Phase 2).** Close the finger around a rigid object and measure how contact forces distribute across the phalanges as a function of object shape and cable tension. Real human fingers have exactly this kind of underactuation, and it's part of why human grasping is so robust. Show the simulated finger doing the same thing, with contact force arrows visualized.

A fourth stretch experiment, only if days 10–11 have slack: **closed-loop cable-tension optimization.** Define a target task ("grasp this object and hold it against gravity"), use Newton's differentiability to optimize a cable-tension trajectory that achieves it. Model-based control through a differentiable simulator. This is the experiment that, if it works, turns the artifact from "physics study + robotics application" into "active research demonstration of what Newton was built for." Drop it without regret if you don't have time — articulate it as future work instead.

### The connecting tissue

The single paragraph that ties the two phases together in the writeup, and the sentence you say to Frey:

*"I started with the spinning closed chain because I'd been thinking about Stephenson's Great Chain, and the cable solver let me validate Newton against analytic baselines cleanly. Once I trusted it, I asked whether the same primitive could simulate the cable-driven actuation that next-generation humanoid hands depend on, so I built a tendon-driven finger and ran the experiments that exercise the hardest cases — frictional routing, slack-to-tension transitions, and underactuated grasping."*

---

## The Soft-Body Angle

Soft-body simulation for robotics is the open frontier the Newton team explicitly cares about (per the NVIDIA PM interview). It's "solved" in graphics but not in robotics — current techniques are visually plausible but not physically accurate or differentiably useful, and Newton's multi-solver architecture exists to attack this gap. Any place you can credibly fold soft-body into the artifact strengthens it for this audience.

Three places it can show up across the two phases:

**Phase 1 stretch — continuous elastic rod for the spinning chain.** Run the steady-state validation in two regimes (discrete rigid links vs continuous elastic rod with bending stiffness) and compare. The interesting question becomes: does the rigid-link chain at large N converge to the elastic-rod limit, or to something subtly different? Only viable if Newton's cable solver supports bending stiffness and closed loops — you find out on day 1. Drop without regret if it doesn't.

**Phase 2 stretch — soft fingertip pad on the tendon-driven finger.** Add a deformable pad at the fingertip (using Newton's deformable solver) so the contact patch with the grasped object is soft, not point-contact. This is a small addition that turns the underactuated grasping experiment into a multi-solver scene — exactly the configuration Newton was built for. Highest-leverage soft-body addition because it's a single localized change with disproportionate signal value.

**Phase 2 future-work — fully soft pneumatic finger.** Don't build this. Articulate it as the natural next step in the writeup: replace the rigid phalanges with a continuous soft body driven by internal pressure or an embedded tendon. This is where Newton's multi-solver story matters most, and naming it explicitly as future work signals that you understand the trajectory.

### The day-one go/no-go questions

After running the examples on day 1, answer:

1. Does Newton's cable/rope solver support bending stiffness (not just inextensible string)?
2. Can it be configured as a closed loop (for the Phase 1 stretch)?
3. Is there a deformable solver that can be coupled to rigid bodies in the same scene (for the Phase 2 stretch)?

Use the answers to decide which soft-body additions are realistic. Default if any of these are hard: skip the stretch, ship the rigid versions cleanly, mention soft-body extensions as future work.

---

## What You Need to Learn (and When)

Do not learn this material front-loaded. Learn it just-in-time, when you hit a specific question. The list below is ordered by when you'll actually need each piece.

### Phase 0 — Before you write any code

- **The Newton README and example list.** Read every README in the `newton-physics` GitHub org. Skim every example file. You're building a map of what exists.
- **Stephenson's acknowledgments section in *Seveneves*** if you can find it. The exact phrasing of his note about chain dynamics not being well-studied is worth quoting in your writeup motivation.
- **The continuous-rope rotating-ring analytic result.** Search "rotating chain" or "rotating closed loop tension" — the classical result is that a uniform spinning closed loop forms a stable circle held by the balance of centripetal acceleration against tangential tension, with tension T = μω²r² (μ = mass per unit length, ω = angular velocity, r = ring radius). Make sure you can derive this on paper before you start. It's a one-page derivation and it's the baseline you'll validate against.

### Phase 1 — When you start building the scene

- **Newton's cable/rope solver API.** Read the source of whichever cable example is closest to a closed loop. Modify it. Read more source.
- **Newton's joint primitives.** You'll need to connect rigid links with constraints that allow flexible bending. Figure out which joint type is right for your case.
- **OpenUSD basics.** Newton uses USD for scene description. You don't need to be an expert — you just need to understand how to construct a scene programmatically.

### Phase 2 — When experiments start

- **Mode analysis for vibrating rings.** When you start characterizing stability modes, you'll want to know what the predicted mode structure is for a rotating elastic ring. Search "rotating ring vibration modes" — there's a classical literature here from mechanical engineering.
- **Numerical integration stability.** If your simulation blows up (which it will, at least once), you'll need to understand why — implicit vs explicit time-stepping, stability criteria, the stiffness of the joint constraints.

### Phase 3 — During the writeup

- **The DiffTaichi paper** (Hu et al. 2020) — for vocabulary about differentiable physics, since your "future work" section will discuss closed-loop differentiable optimization of cable-tension trajectories.
- **One or two recent papers from the DeepMind robotics simulation group** — pick from Frey's coauthor list. You want to be able to reference his team's work in your writeup.
- **Cable-driven and tendon-driven mechanism analysis.** When you build the finger, you'll need the textbook force-balance equations for tendons routed through via points: the joint torque from a tendon is the cable tension times the moment arm at each via point, summed across all joints the tendon crosses. Search "tendon-driven mechanism statics" or look at any classical robotic-hand paper (Shadow Hand, ACT Hand, Stanford manipulator papers). This is your analytic baseline for Phase 2 experiment 1.

---

## Two-Week Plan

### Week 1 — Phase 1: The Great Chain

**Day 1 (today): install, explore, scope.**
- Install Newton: `pip install "newton[examples]"`
- Run `python -m newton.examples basic_pendulum` — verify the toolchain works.
- Run every example. For each one, watch it in the viewer, then read the source top to bottom.
- **Specifically investigate the cable/rope/rod examples.** Answer: (1) does any solver support bending stiffness? (2) can it be closed into a loop? (3) can it be coupled to rigid bodies in the same scene? — the third question is critical for Phase 2.
- Goal: a working install, a mental map of the API, and confidence that Phase 2's tendon-driven finger is buildable on Newton's primitives.

**Day 2: pick the closest example and modify it.**
- Find the cable/rope example closest to a closed loop.
- Make trivial modifications: change masses, change link counts, change initial conditions. Watch what happens.
- Goal: you understand how scenes are constructed, how the simulation loop runs, how to read state out.

**Day 3: closed loop, N=4 → parametrized.**
- Build the simplest possible closed loop of 4 links connected end-to-end.
- Give it an initial angular velocity around the center.
- Refactor so N is a parameter; test at N = 4, 8, 16, 32.
- Add basic instrumentation: position of each link over time, centroid, deviation from circular shape.
- Goal: a parametrizable spinning closed chain that produces clean data.

**Day 4: validate and sweep.**
- For N=32, spin at known angular velocity, measure steady-state radius and tension, compare to T = μω²r². Should agree within a few percent.
- Then run the sweep: N = 2, 4, 8, 16, 32, 64, 128. Record steady-state shape and tension distribution at each.
- Goal: experiment 1 (validation) and experiment 2 (interpolation) data are both collected.

**Day 5: one stability observation + Phase 1 wrap.**
- Take the validated steady-state ring at one rotation rate. Perturb it (radial pulse). Watch what mode develops. Record one interesting observation.
- Generate the Phase 1 figures: validation plot, headline interpolation grid, perturbation visualization.
- **Phase 1 is now done.** Stop. Do not extend it further.
- Goal: Phase 1 has a clean writeup-ready set of results and figures.

### Week 2 — Phase 2: The Tendon-Driven Finger + Writeup

**Day 6: build the finger scene.**
- Three rigid links (phalanges), two pin joints between them, a passive return spring at each joint.
- One Newton cable routed from a fixed anchor at the base, through via points or sheaths at each joint, to the fingertip.
- Verify: pulling the cable curls the finger; releasing it lets the springs straighten it.
- Goal: a working tendon-driven finger you can drive interactively.

**Day 7: force transmission validation.**
- Apply known cable tensions in static equilibrium against an external load.
- Measure joint angles and contact forces.
- Compare to the analytic textbook prediction for tendon-driven mechanisms (you'll need to derive this on paper — it's the classical force-balance through moment arms at each via point).
- Goal: experiment 1 of Phase 2 is done. You know whether Newton's cable solver is force-accurate.

**Day 8: slack-to-tension experiment.**
- Drive the cable in a sinusoidal pattern that takes it through the slack regime each cycle.
- Attempt to backprop through the simulation. Record where gradients are well-behaved and where they pathologize.
- Plot gradient norms across the cycle. Mark the slack-to-tension transition.
- Goal: experiment 2 of Phase 2 is done — and this is the result Frey will read most carefully.

**Day 9: underactuated grasping (headline visualization).**
- Place a rigid object near the finger. Close the finger around it by pulling the cable.
- Measure how contact forces distribute across the phalanges as a function of object shape and cable tension.
- Visualize the grasp with contact force arrows.
- Goal: experiment 3 of Phase 2 is done. The headline image of the artifact exists.

**Day 10: catch-up + stretch experiment if time.**
- Phase 2 catch-up day. Something will have eaten more time than planned.
- If everything is on track, attempt experiment 4: closed-loop cable-tension optimization for a target grasp using Newton's differentiability. If you don't get it working in one day, drop it without regret and articulate it as future work.
- Goal: Phase 2 is done. Drop the stretch experiment cleanly if needed.

**Day 11: figures and writeup draft.**
- Polish all figures from both phases: chain validation plot, interpolation grid, finger force transmission plot, slack gradient plot, grasping visualization.
- Write the technical note (4–5 pages, see writeup outline below).
- Goal: a complete first draft of the writeup with all figures embedded.

**Day 12: polish and self-critique.**
- Re-read the note as if you were Frey. Cut anything that's filler. Make every sentence earn its place.
- Make sure the "what Newton needs" section is honest and specific, not generic.
- Make sure the connecting paragraph between Phase 1 and Phase 2 reads as a real arc, not a forced bridge.

**Day 13: CV update and message draft.**
- Add the project to the CV.
- Draft the message to Frey, with the writeup attached.
- Don't send it yet.

**Day 14: send.**
- Send the writeup and message to Frey via the personal channel.
- Buffer day for last polish or last-minute fixes.

---

## Writeup Outline

A single 4–5 page document with two parts and connecting tissue.

### Section 1 — Motivation and Arc (½ page)

A paragraph on the Great Chain in *Seveneves* and Stephenson's note about chain dynamics being under-studied. A paragraph on why this is interesting now: Newton's cable solver is the first widely available tool that makes this kind of investigation tractable. A short paragraph on the arc: started with the spinning chain because of curiosity, validated Newton on it, then asked whether the same primitive could solve the cable-driven manipulation problem that next-generation humanoid hands depend on.

### Section 2 — Phase 1: Validating Newton on Spinning Closed Chains (1.5 pages)

- **Steady-state validation.** Setup, analytic prediction (T = μω²r²), simulation result, agreement plot.
- **Discrete-to-continuous interpolation.** The headline grid figure across N. A paragraph on what changes as N increases.
- **One stability observation.** What you perturbed, what mode appeared. Brief, honest.

### Section 3 — Phase 2: Tendon-Driven Manipulation (1.5–2 pages)

The arc paragraph: "Once I trusted the cable solver on the spinning chain, I wanted to test it on the simulation problem that matters most for current robotics — cable-driven dexterous manipulation. Almost every serious humanoid hand is tendon-driven, and accurately simulating these systems is an open problem because cables interact with their routing surfaces through frictional contact, slack-to-tension transitions produce gradient pathologies, and shared tendons couple multiple joints non-trivially."

- **Force transmission validation.** Static equilibrium, analytic prediction from textbook tendon-mechanism analysis, agreement plot.
- **Slack-to-tension gradient pathology.** Sinusoidal cable drive, gradient norm plot across the slack transition, characterization of where Newton's gradients are well-behaved.
- **Underactuated grasping.** The headline visualization for Phase 2: a finger conforming to an object with contact force arrows. Distribution of contact forces across phalanges as a function of object shape.

Be concrete. Numbers, plots, screenshots. No adjectives.

### Section 4 — Where Newton Worked and Where It Didn't (½ page)

The honest section. Combined feedback from both phases. Specific things that worked well, specific things that were rough or required workarounds. Things you wished the API had. *This section is the most valuable to Frey.* Be specific and constructive.

### Section 5 — What This Enables and What I'd Do Next (½ page)

One paragraph framing this as the simplest case of a much larger class of cable-driven manipulation problems — multi-finger hands, coupled tendon networks, soft fingertip pads, fully soft pneumatic fingers (the multi-solver case Newton was built for). One paragraph on the natural next step: closed-loop differentiable optimization of cable-tension trajectories for dexterous tasks. One sentence as the implicit ask: this is the kind of work I'd love to do more of with your team this summer.

---

## Resources to Pull In When You Need Them

In rough order of when you'll want each one:

- **Newton GitHub (newton-physics org).** README, examples, issues, source. This is your primary documentation. Spend more time here than anywhere else.
- **Newton Discord or Discussions tab if it exists.** Real-time questions when you're stuck.
- **MuJoCo "Computation" documentation.** Background on how production physics simulators handle constraints, contacts, and integration. Newton's rigid body backend is MuJoCo Warp, so this is the right reference.
- **Classical mechanics references on rotating chain dynamics.** Search terms: "rotating chain tension," "spinning closed loop stability," "rotating ring vibration modes."
- **DiffTaichi paper (Hu et al. 2020).** For vocabulary in your future-work section about differentiable physics.
- **One recent Frey-coauthored paper.** Pick from his Google Scholar page. Read it well enough to reference it intelligently.

---

## Ground Rules

A few things to internalize before starting.

**Tolerate confusion.** You will spend most of week 1 not understanding things. That's the work. Don't avoid it by reading more tutorials — push through it by modifying code and seeing what happens.

**Use AI coding assistance aggressively, but understand every line.** When Claude writes code for you, read it, modify it, break it, fix it. The goal is fluency in the system, not throughput.

**Don't run the closed-loop cable-tension optimization unless Phase 2 is on track.** It's the stretch experiment for day 10 only. Trying to do it earlier or pushing it past day 10 will eat the writeup time. Drop it without regret if needed and articulate it as future work — the future-work hook is itself valuable.

**Stop Phase 1 on day 5, no matter what.** The spinning chain is a means to an end (validating the cable solver and giving you fluency). Don't let it sprawl into experiments you didn't plan. The real artifact is the two phases together, and Phase 2 is where the value lands for Frey.

**Don't add scope.** You will be tempted to simulate the full Eye, model the docking dynamics, add multiple coupled chains. Don't. The artifact is stronger if it does one thing cleanly.

**Plot everything.** Numbers in tables are forgettable. Plots are remembered. Every experiment gets at least one figure.

**Write the writeup as if Frey is reading it tomorrow.** Because he is.

---

## The One-Sentence Pitch for the Conversation

"I started with the spinning closed chain because I'd been thinking about Stephenson's Great Chain in *Seveneves* — Stephenson actually flagged spinning chain dynamics as under-studied in the acknowledgments — and the cable solver let me validate Newton against the analytic continuous-rope prediction cleanly. Once I trusted it, I asked whether the same primitive could simulate the cable-driven actuation that next-generation humanoid hands depend on, so I built a tendon-driven finger and ran the experiments that exercise the hardest cases: frictional cable routing, slack-to-tension gradient pathologies, and underactuated grasping. I have a writeup of where Newton's cable solver works and where it's still rough across both, and a clear next step toward closed-loop differentiable optimization of cable-tension trajectories — which I think is the kind of thing Newton is uniquely positioned to enable. Want to take a look?"

That sentence is the goal of the next two weeks. It threads curiosity → validation → application → research direction in one breath. Build the thing that lets you say it truthfully.

---

*Start tonight. Install Newton. Run basic_pendulum. Don't read any more roadmap documents — including this one — until tomorrow morning.*
