// Runs the original A.I. Wars web engine (engine_core.js + engine_caicl.js) headlessly
// and dumps a per-tick state trace, for conformance testing of the pgx port.
//
// Usage: node run_ref.js <battle.json> [out.json]   (AIWARS_SRC: dir holding the engine .js files)
//
// battle.json:
//   { "map_text": "...", "node": {"x":..,"y":..} | null (disabled),
//     "settings": {...},             // optional engine settings
//     "cybugs": [{"slot":1, "script_text":"...", "iff":"41001", "team":0, "x":..,"y":..,"h":..}],
//     "tape_file": "tape.bin",       // uint32 LE, shape [max_turns, n_bugs, n_draws]
//     "n_draws": 32,
//     "max_turns": 2000,
//     "trace_vars": true }
//
// Randomness: inside a slot's turn the k-th Math.random() call returns
// tape[tick][bug_index][k] / 2^24. Outside slot turns (map loading) it returns 0.5;
// the battle spec pins the node and every start position and heading, so those
// draws are unused.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SRC = process.env.AIWARS_SRC || path.join(__dirname, "..");
const ctx = { console, Math: Object.create(Math) };
ctx.globalThis = ctx;
vm.createContext(ctx);
for (const f of ["engine_core.js", "engine_caicl.js"]) {
  vm.runInContext(fs.readFileSync(path.join(SRC, f), "utf8"), ctx, { filename: f });
}

const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const out = process.argv[3];
const nBugs = spec.cybugs.length;
const nDraws = spec.n_draws || 32;
let tape = null;
if (spec.tape_file) {
  const buf = fs.readFileSync(path.resolve(path.dirname(process.argv[2]), spec.tape_file));
  tape = new Uint32Array(buf.buffer, buf.byteOffset, buf.length / 4);
}

const rng = { tick: -1, bug: -1, k: 0, overflow: 0 };
ctx.Math.random = function () {
  if (rng.tick < 0 || !tape) return 0.5;
  if (rng.k >= nDraws) { rng.overflow++; rng.k = 0; }
  const v = tape[(rng.tick * nBugs + rng.bug) * nDraws + rng.k];
  rng.k++;
  return v / 16777216;
};

const AC = ctx.AIWARS_CORE, CAICL = ctx.CAICL;
const settings = Object.assign({}, spec.settings || {});
if (spec.max_turns) settings.stop_at_click = spec.max_turns;
const cfg = {
  map_text: spec.map_text,
  map_name: "test",
  settings,
  node: spec.node ? { fixed: true, x: spec.node.x, y: spec.node.y } : null,
  cybugs: spec.cybugs.map((b) => ({
    slot: b.slot, name: "Bug" + b.slot, filename: "b.ai", iff: b.iff, team: b.team || 0,
    script_text: b.script_text, x: b.x, y: b.y, h: b.h,
  })),
};
if (!spec.node) settings.strategy_node_disabled = true;

AC.initBattle(cfg);
const G = AC.getState();
const slots = G.active_slots;

let maxNest = 0, maxSub = 0;
const trace = [];
function snap() {
  const bugs = slots.map((s) => {
    const o = {
      x: G.Cybug_x[s], y: G.Cybug_y[s], h: G.Cybug_h[s], damage: G.Cybug_damage[s],
      fuel: G.Cybug_fuel[s], ammo: G.Cybug_ammo[s], score: G.Cybug_score[s],
      shield: G.Cybug_shield[s], overheat: G.Cybug_overheat[s], penalty: G.Cybug_penalty[s],
      cloak: G.Cybug_Cloaked[s] ? 1 : 0, mine: G.Cybug_mine[s], dead: G.Cybug_dead[s],
      burn: G.Cybug_burn[s], fuel_rate: G.Cybug_fuel_rate[s], queen: G.Cybug_queen[s] ? 1 : 0,
      mistat: G.Cybug_mistat[s], misx: G.Cybug_misx[s], misy: G.Cybug_misy[s],
      grstat: G.Cybug_grstat[s], grax: G.Cybug_grax[s], gray: G.Cybug_gray[s],
      grad: G.Cybug_grad[s], grat: G.Cybug_grat[s],
      iff: G.Cybug_iff[s], scan: G.Cybug_scan[s], random: G.Cybug_random[s],
      barrier: G.barrier[s], mem: G.Cybug_mem[s], nest: G.nestlevel[s], sub: G.curnest[s],
    };
    if (spec.trace_vars) {
      const v = {};
      for (const k of Object.keys(G.Cybug_var[s])) {
        const val = G.Cybug_var[s][k];
        if (val !== "" && val !== undefined) v[k] = val;
      }
      o.vars = v;
    }
    return o;
  });
  let grid = "";
  for (let r = 0; r < 32; r++) for (let c = 0; c < 45; c++) {
    const v = G.map_grid[c][r];
    grid += v === 0 ? "." : v === 20 ? "#" : v === 30 ? "F" : v === 40 ? "9" : String.fromCharCode(48 + v);
  }
  return { t: G.cur_turn, mode: G.battle_mode, movecount: G.movecount, bugs, grid };
}

trace.push(snap());
while (G.battle_mode === 1) {
  AC.startTurn();
  for (let si = 0; si < slots.length; si++) {
    const slot = slots[si];
    rng.tick = G.cur_turn; rng.bug = si; rng.k = 0;
    if (AC.slotPreProcess(slot)) CAICL.executeTurn(slot);
    rng.tick = -1;
    AC.checkDead(slot);
    AC.slotPostProcess(slot);
    AC.checkDead(slot);
    maxNest = Math.max(maxNest, G.nestlevel[slot]);
    maxSub = Math.max(maxSub, G.curnest[slot]);
  }
  AC.endTurn();
  trace.push(snap());
}

const result = {
  winner: G.winner, ticks: G.cur_turn,
  scores: slots.map((s) => G.Cybug_score[s]),
  max_nest: maxNest, max_sub: maxSub, rng_overflow: rng.overflow,
  trace,
};
const txt = JSON.stringify(result);
if (out) fs.writeFileSync(out, txt); else process.stdout.write(txt);
