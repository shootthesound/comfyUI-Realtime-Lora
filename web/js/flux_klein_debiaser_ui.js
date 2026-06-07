import { app } from "../../../scripts/app.js";

// ============================================================================
// DIT Deep Debiaser â€” FLUX.2 Klein UI
//
// Architecture-verified color scheme:
//   Blue   (#4488ff) = img stream (double blocks, SEPARATE)
//   Purple (#aa66dd) = txt stream (double blocks, SEPARATE)
//   Orange (#dd8833) = single blocks (JOINT cross-modal)
//   Green  (#66aa66) = global / timing / final
// ============================================================================

const N_DOUBLE = 8;
const N_SINGLE = 24;
const DB_SUBS = ["img_attn", "img_mlp", "txt_attn", "txt_mlp"];

const GLOBAL_BLOCKS = [
    "img_in", "txt_in", "time_in", "db_mod_img", "db_mod_txt", "sb_mod", "final_layer",
];
const DB_BLOCKS = [];
for (let i = 0; i < N_DOUBLE; i++)
    for (const sub of DB_SUBS) DB_BLOCKS.push(`db${i}_${sub}`);
const SB_BLOCKS = [];
for (let i = 0; i < N_SINGLE; i++) SB_BLOCKS.push(`sb${i}`);
const ALL_BLOCKS = [...GLOBAL_BLOCKS, ...DB_BLOCKS, ...SB_BLOCKS];

// --- Colors ---
function blockColor(n) {
    if (n === "img_in" || n === "db_mod_img") return "#4488ff";
    if (n === "txt_in" || n === "db_mod_txt") return "#aa66dd";
    if (n === "time_in" || n === "final_layer") return "#66aa66";
    if (n === "sb_mod") return "#dd8833";
    if (n.includes("_img_attn")) return "#4488ff";
    if (n.includes("_img_mlp")) return "#3377dd";
    if (n.includes("_txt_attn")) return "#aa66dd";
    if (n.includes("_txt_mlp")) return "#9955cc";
    if (n.startsWith("sb")) return "#dd8833";
    return "#888";
}

function blockBg(n, on) {
    if (!on) return "#1a1a1a";
    if (n.includes("_img_") || n === "img_in" || n === "db_mod_img") return "#1c2030";
    if (n.includes("_txt_") || n === "txt_in" || n === "db_mod_txt") return "#241c2e";
    if (n.startsWith("sb") || n === "sb_mod") return "#2a2214";
    return "#252525";
}

// --- Preset data (mirrors Python) ---
function makePreset(fn) {
    const r = {};
    for (const b of ALL_BLOCKS) {
        const o = fn(b);
        if (o) r[b] = o;
    }
    return r;
}

function sbIdx(b) { return parseInt(b.slice(2)); }

const PRESETS = {
    "Custom": null,
    "Default": makePreset(() => ({ e: true, s: 1.0 })),

    "Weaken All Singles 95%": makePreset(b =>
        b.startsWith("sb") && !b.includes("mod") ? { e: true, s: 0.95 } : null),
    "Weaken All Singles 90%": makePreset(b =>
        b.startsWith("sb") && !b.includes("mod") ? { e: true, s: 0.90 } : null),
    "Weaken All Singles 85%": makePreset(b =>
        b.startsWith("sb") && !b.includes("mod") ? { e: true, s: 0.85 } : null),
    "Weaken Late Singles 90% (SB12-23)": makePreset(b =>
        b.startsWith("sb") && !b.includes("mod") && sbIdx(b) >= 12 ? { e: true, s: 0.90 } : null),
    "Weaken Late Singles 85% (SB12-23)": makePreset(b =>
        b.startsWith("sb") && !b.includes("mod") && sbIdx(b) >= 12 ? { e: true, s: 0.85 } : null),
    "Weaken Early Singles 90% (SB0-7)": makePreset(b =>
        b.startsWith("sb") && !b.includes("mod") && sbIdx(b) < 8 ? { e: true, s: 0.90 } : null),

    "Weaken DB img_mlp 90%": makePreset(b =>
        b.includes("_img_mlp") ? { e: true, s: 0.90 } : null),
    "Weaken DB txt_mlp 90%": makePreset(b =>
        b.includes("_txt_mlp") ? { e: true, s: 0.90 } : null),
    "Weaken DB img_attn 90%": makePreset(b =>
        b.includes("_img_attn") ? { e: true, s: 0.90 } : null),
    "Weaken DB txt_attn 90%": makePreset(b =>
        b.includes("_txt_attn") ? { e: true, s: 0.90 } : null),
    "Weaken ALL img stream 90%": makePreset(b => {
        if (b === "img_in" || b === "db_mod_img") return { e: true, s: 0.90 };
        if (b.includes("_img_")) return { e: true, s: 0.90 };
        return null;
    }),
    "Weaken ALL txt stream 90%": makePreset(b => {
        if (b === "txt_in" || b === "db_mod_txt") return { e: true, s: 0.90 };
        if (b.includes("_txt_")) return { e: true, s: 0.90 };
        return null;
    }),

    "Boost img_in 115%": makePreset(b =>
        b === "img_in" ? { e: true, s: 1.15 } : null),
    "Boost txt_in 115%": makePreset(b =>
        b === "txt_in" ? { e: true, s: 1.15 } : null),
    "Weaken sb_mod 90%": makePreset(b =>
        b === "sb_mod" ? { e: true, s: 0.90 } : null),
    "Global 95%": makePreset(() => ({ e: true, s: 0.95 })),
    "Global 90%": makePreset(() => ({ e: true, s: 0.90 })),

    "Protect Reference (edit mode)": makePreset(b => {
        if (b.startsWith("sb") && !b.includes("mod") && sbIdx(b) >= 12)
            return { e: true, s: 0.88 };
        if (b === "img_in") return { e: true, s: 1.10 };
        return null;
    }),
    "Stronger Prompt (edit mode)": makePreset(b => {
        if (b === "txt_in") return { e: true, s: 1.15 };
        if (b === "db_mod_txt") return { e: true, s: 1.10 };
        if (b.includes("_txt_attn") || b.includes("_txt_mlp"))
            return { e: true, s: 1.10 };
        return null;
    }),
};

// ============================================================================
// EXTENSION
// ============================================================================

app.registerExtension({
    name: "FluxKleinDeepDebiaser.UI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FluxKleinDeepDebiaser") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                if (node._ddbInit) return;
                node._ddbInit = true;
                node._setupCombinedWidgets();
                node._setupPreset();
                if (node.size[0] < 520) node.size[0] = 520;
                node.setDirtyCanvas(true);
            }, 50);
        };

        const origConf = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origConf) origConf.apply(this, arguments);
            setTimeout(() => this.setDirtyCanvas(true), 120);
        };

        const origExec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (out) {
            if (origExec) origExec.apply(this, arguments);
            if (out?.analysis_json?.[0]) {
                try { this._ad = JSON.parse(out.analysis_json[0]); } catch {}
                this.setDirtyCanvas(true);
            }
        };

        // ----- Combined checkbox + slider -----
        nodeType.prototype._setupCombinedWidgets = function () {
            const strNames = new Set(
                this.widgets.filter(w => w.name.endsWith("_str")).map(w => w.name)
            );

            for (const w of this.widgets) {
                const sn = w.name + "_str";
                if (!strNames.has(sn)) continue;
                const sw = this.widgets.find(ww => ww.name === sn);
                if (!sw) continue;
                this._makeCombined(w, sw, w.name);
            }

            this.setSize(this.computeSize());
        };

        nodeType.prototype._makeCombined = function (toggle, strength, name) {
            const M = 8, CB = 13, GAP = 5, LW = 155, VW = 36;
            const MIN = -2.0, MAX = 2.0, STEP = 0.05;

            toggle.draw = function (ctx, node, ww, y, wh) {
                const SW = ww - M - CB - GAP - LW - GAP - VW - M - GAP;
                const cbX = M;
                const labelX = cbX + CB + GAP;
                const sliderX = labelX + LW + GAP;

                const on = Boolean(toggle.value);
                let str = parseFloat(strength.value);
                if (isNaN(str)) str = 1.0;

                const col = blockColor(name);

                // Row bg
                ctx.fillStyle = blockBg(name, on);
                ctx.fillRect(0, y, ww, wh);

                // Checkbox
                ctx.strokeStyle = on ? col : "#555";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(cbX, y + (wh - CB) / 2, CB, CB);
                if (on) {
                    ctx.fillStyle = col;
                    ctx.fillRect(cbX + 2, y + (wh - CB) / 2 + 2, CB - 4, CB - 4);
                }

                // Label
                ctx.globalAlpha = on ? 1.0 : 0.35;
                ctx.fillStyle = on ? "#ddd" : "#666";
                ctx.font = "11px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                let lbl = name;
                while (ctx.measureText(lbl + "â€¦").width > LW - 4 && lbl.length > 4)
                    lbl = lbl.slice(0, -1);
                if (lbl !== name) lbl += "â€¦";
                ctx.fillText(lbl, labelX, y + wh / 2);
                ctx.globalAlpha = 1.0;

                // Slider track
                const ty = y + wh / 2, th = 4;
                const range = MAX - MIN;
                const norm = (str - MIN) / range;
                const zNorm = (0 - MIN) / range;

                ctx.fillStyle = "#333";
                ctx.beginPath();
                ctx.roundRect(sliderX, ty - th / 2, SW, th, 2);
                ctx.fill();

                if (on) {
                    const zX = sliderX + zNorm * SW;
                    const sX = sliderX + norm * SW;
                    ctx.fillStyle = str >= 0 ? col : "#ff6655";
                    ctx.beginPath();
                    ctx.roundRect(Math.min(zX, sX), ty - th / 2, Math.abs(sX - zX), th, 2);
                    ctx.fill();
                }

                // Thumb
                ctx.fillStyle = on ? "#fff" : "#555";
                ctx.beginPath();
                ctx.arc(sliderX + norm * SW, ty, 5, 0, Math.PI * 2);
                ctx.fill();

                // Value
                ctx.fillStyle = on ? "#ccc" : "#555";
                ctx.textAlign = "right";
                ctx.font = "10px monospace";
                ctx.fillText(str.toFixed(2), ww - M, y + wh / 2);
            };

            const origMouse = toggle.mouse?.bind(toggle);
            toggle.mouse = function (event, pos, node) {
                const ww = node.size[0];
                const SW = ww - M - CB - GAP - LW - GAP - VW - M - GAP;
                const sliderX = M + CB + GAP + LW + GAP;
                const valueX = ww - M - VW;  // left edge of the numeric value display
                const lx = pos[0];

                // Click on the numeric value column → prompt for direct entry.
                // Without this the click falls through to the boolean toggle handler
                // (which is what was making clicks "disable" the row in #42).
                if (event.type === "pointerdown" && lx >= valueX - 4) {
                    const current = parseFloat(strength.value);
                    const seed = isNaN(current) ? "1.0" : current.toFixed(2);
                    const input = window.prompt(
                        `Set ${name} strength (${MIN.toFixed(1)} to ${MAX.toFixed(1)}):`,
                        seed
                    );
                    if (input !== null && input !== "") {
                        const parsed = parseFloat(input);
                        if (!isNaN(parsed)) {
                            let v = Math.max(MIN, Math.min(MAX, parsed));
                            v = Math.round(v / STEP) * STEP;
                            strength.value = v;
                            node.setDirtyCanvas(true);
                        }
                    }
                    return true;
                }

                // Slider drag — RELATIVE / anchored. Grabbing the track anchors at the
                // current value instead of jumping to the cursor (the jump-to-click +
                // drag-all-the-way-to-the-edge behaviour was the #42 complaint). You then
                // adjust from where you grabbed, so small drags give fine control and you
                // never have to drag to the track edge to reach min/max.
                if (event.type === "pointerdown" && lx >= sliderX - 4 && lx <= sliderX + SW + 4) {
                    toggle._strDrag = { x: lx, v: parseFloat(strength.value) || 0 };
                    return true;
                }
                if (event.type === "pointermove" && toggle._strDrag) {
                    if (event.buttons === 0) {
                        toggle._strDrag = null;            // button released off-widget
                    } else {
                        let v = toggle._strDrag.v + ((lx - toggle._strDrag.x) / SW) * (MAX - MIN);
                        v = Math.round(v / STEP) * STEP;
                        v = Math.max(MIN, Math.min(MAX, v));
                        strength.value = v;
                        node.setDirtyCanvas(true);
                        return true;
                    }
                }
                if (event.type === "pointerup" && toggle._strDrag) {
                    toggle._strDrag = null;
                    return true;
                }

                if (origMouse) return origMouse(event, pos, node);
                return false;
            };

            // Hide the strength widget row
            strength.draw = function () {};
            strength.computeSize = function () { return [0, -4]; };
        };

        // ----- Presets -----
        nodeType.prototype._setupPreset = function () {
            const node = this;
            const pw = this.widgets.find(w => w.name === "preset");
            if (!pw) return;
            pw.options = pw.options || {};
            pw.options.values = Object.keys(PRESETS);
            if (!Object.keys(PRESETS).includes(pw.value)) pw.value = "Default";
            const origCb = pw.callback;
            pw.callback = function (value) {
                node._applyPreset(value);
                if (origCb) origCb.call(this, value);
            };
            // Always save as Custom so reloading doesn't re-trigger
            pw.serializeValue = function () { return "Custom"; };
        };

        nodeType.prototype._applyPreset = function (name) {
            const preset = PRESETS[name];
            if (!preset) return;
            for (const b of ALL_BLOCKS) {
                const tw = this.widgets.find(w => w.name === b);
                const sw = this.widgets.find(w => w.name === b + "_str");
                const ov = preset[b];
                if (ov) {
                    if (tw) tw.value = ov.e;
                    if (sw) sw.value = ov.s;
                } else {
                    if (tw) tw.value = true;
                    if (sw) sw.value = 1.0;
                }
            }
            this.setDirtyCanvas(true);
        };
    }
});
