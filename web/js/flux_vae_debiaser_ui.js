import { app } from "../../../scripts/app.js";

// ============================================================================
// FLUX VAE DEEP DEBIASER — 125 Individual Tensor Control UI
// For Flux 2 Klein 9B's VAE (32ch latent)
// Every conv, norm, shortcut, q/k/v/proj gets its own checkbox+slider.
// ============================================================================

function getImpactColor(score) {
    if (score < 10) return "#0066ff";
    if (score < 20) return "#0088ff";
    if (score < 30) return "#00aaff";
    if (score < 40) return "#00cccc";
    if (score < 50) return "#00cc66";
    if (score < 60) return "#88cc00";
    if (score < 70) return "#cccc00";
    if (score < 80) return "#ff9900";
    if (score < 90) return "#ff6600";
    return "#ff3300";
}

function getSubColor(name) {
    if (name === "bn") return "#777777";
    if (name.includes("pqconv") || name.includes("qconv")) return "#aa88cc";
    if (name.includes("attn_q")) return "#4488ff";
    if (name.includes("attn_k")) return "#5599ff";
    if (name.includes("attn_v")) return "#66aaff";
    if (name.includes("attn_proj")) return "#3377ee";
    if (name.includes("attn_norm")) return "#6699cc";
    if (name.includes("norm_out")) return "#cccc66";
    if (name.includes("conv_out")) return "#88cccc";
    if (name.includes("conv_in")) return "#88cc88";
    if (name.includes("nin_shortcut")) return "#cc88cc";
    if (name.includes("_up")) return "#cc66cc";  // upsample/downsample
    if (name.includes("_down")) return "#cc66cc";
    if (name.includes("norm")) return "#99bb66";
    if (name.includes("conv1")) return "#ff8855";
    if (name.includes("conv2")) return "#ff6633";
    if (name.startsWith("d_")) return "#ff7744";
    if (name.startsWith("e_")) return "#4488ff";
    return "#5599ff";
}

function getSubBg(name, enabled) {
    if (!enabled) return "#1e1e1e";
    if (name.startsWith("d_u0")) return "#2d2525";
    if (name.startsWith("d_u1")) return "#2d2828";
    if (name.startsWith("d_u2") || name.startsWith("d_u3")) return "#2d2a28";
    if (name.startsWith("d_mid")) return "#28282d";
    if (name.startsWith("e_d0") || name.startsWith("e_d1")) return "#25282d";
    if (name.startsWith("e_d2") || name.startsWith("e_d3")) return "#28282d";
    if (name.startsWith("e_mid")) return "#282d28";
    return "#2a2a2a";
}

// ============================================================================
// Block definitions — MUST match Python BLOCKS exactly
// ============================================================================

const ALL_BLOCKS = ["bn"];

// Decoder
ALL_BLOCKS.push("d_pqconv", "d_conv_in");
for (const sub of ["q", "k", "v", "proj_out", "norm"]) ALL_BLOCKS.push(`d_mid_attn_${sub}`);
for (const blk of [1, 2]) {
    for (const sub of ["conv1", "conv2", "norm1", "norm2"]) ALL_BLOCKS.push(`d_mid_b${blk}_${sub}`);
}
const DEC_NIN = new Set(["0_0", "1_0"]);
for (let stage = 0; stage < 4; stage++) {
    for (let b = 0; b < 3; b++) {
        for (const sub of ["conv1", "conv2", "norm1", "norm2"]) ALL_BLOCKS.push(`d_u${stage}b${b}_${sub}`);
        if (DEC_NIN.has(`${stage}_${b}`)) ALL_BLOCKS.push(`d_u${stage}b${b}_nin_shortcut`);
    }
    if ([1, 2, 3].includes(stage)) ALL_BLOCKS.push(`d_u${stage}_up`);
}
ALL_BLOCKS.push("d_norm_out", "d_conv_out");

// Encoder
ALL_BLOCKS.push("e_conv_in");
const ENC_NIN = new Set(["1_0", "2_0"]);
for (let stage = 0; stage < 4; stage++) {
    for (let b = 0; b < 2; b++) {
        for (const sub of ["conv1", "conv2", "norm1", "norm2"]) ALL_BLOCKS.push(`e_d${stage}b${b}_${sub}`);
        if (ENC_NIN.has(`${stage}_${b}`)) ALL_BLOCKS.push(`e_d${stage}b${b}_nin_shortcut`);
    }
    if ([0, 1, 2].includes(stage)) ALL_BLOCKS.push(`e_d${stage}_down`);
}
for (const sub of ["q", "k", "v", "proj_out", "norm"]) ALL_BLOCKS.push(`e_mid_attn_${sub}`);
for (const blk of [1, 2]) {
    for (const sub of ["conv1", "conv2", "norm1", "norm2"]) ALL_BLOCKS.push(`e_mid_b${blk}_${sub}`);
}
ALL_BLOCKS.push("e_norm_out", "e_conv_out", "e_qconv");

// ============================================================================
// Presets
// ============================================================================

function makePreset(enabled, strength, overrideFn) {
    const overrides = {};
    if (overrideFn) {
        for (const b of ALL_BLOCKS) {
            const val = overrideFn(b);
            if (val !== null) overrides[b] = val;
        }
    }
    return { enabled, strength, overrides };
}

const DEEP_PRESETS = {
    "Custom": null,
    "Default": makePreset("ALL", 1.0, null),
    "Dec All Up0 convs 90%": makePreset("ALL", 1.0, b => (b.startsWith("d_u0") && b.includes("conv")) ? 0.90 : null),
    "Dec All Up0 convs 85%": makePreset("ALL", 1.0, b => (b.startsWith("d_u0") && b.includes("conv")) ? 0.85 : null),
    "Dec All Up0+Up1 convs 90%": makePreset("ALL", 1.0, b => ((b.startsWith("d_u0") || b.startsWith("d_u1")) && b.includes("conv")) ? 0.90 : null),
    "Dec All Up convs 90%": makePreset("ALL", 1.0, b => (b.match(/^d_u\d/) && b.includes("conv")) ? 0.90 : null),
    "Dec All Up convs 85%": makePreset("ALL", 1.0, b => (b.match(/^d_u\d/) && b.includes("conv")) ? 0.85 : null),
    "Dec All Up norms 90%": makePreset("ALL", 1.0, b => (b.match(/^d_u\d/) && b.includes("norm")) ? 0.90 : null),
    "Dec Mid Attn all 90%": makePreset("ALL", 1.0, b => b.startsWith("d_mid_attn") ? 0.90 : null),
    "Dec Mid Attn all 80%": makePreset("ALL", 1.0, b => b.startsWith("d_mid_attn") ? 0.80 : null),
    "Dec Mid Attn Q+K 90%": makePreset("ALL", 1.0, b => (b === "d_mid_attn_q" || b === "d_mid_attn_k") ? 0.90 : null),
    "Dec All norms 90%": makePreset("ALL", 1.0, b => (b.startsWith("d_") && b.includes("norm")) ? 0.90 : null),
    "Dec All norms 85%": makePreset("ALL", 1.0, b => (b.startsWith("d_") && b.includes("norm")) ? 0.85 : null),
    "Dec Up0 convs boost 110%": makePreset("ALL", 1.0, b => (b.startsWith("d_u0") && b.includes("conv")) ? 1.10 : null),
    "Dec Up0 convs boost 120%": makePreset("ALL", 1.0, b => (b.startsWith("d_u0") && b.includes("conv")) ? 1.20 : null),
    "Dec NormOut boost 110%": makePreset("ALL", 1.0, b => b === "d_norm_out" ? 1.10 : null),
    "Dec ConvOut boost 110%": makePreset("ALL", 1.0, b => b === "d_conv_out" ? 1.10 : null),
    "All Decoder 95%": makePreset("ALL", 1.0, b => b.startsWith("d_") ? 0.95 : null),
    "All Decoder 90%": makePreset("ALL", 1.0, b => b.startsWith("d_") ? 0.90 : null),
    "All Decoder 85%": makePreset("ALL", 1.0, b => b.startsWith("d_") ? 0.85 : null),
    "Global 95%": makePreset("ALL", 0.95, null),
    "Global 90%": makePreset("ALL", 0.90, null),
};

// ============================================================================
// Extension registration
// ============================================================================

app.registerExtension({
    name: "FluxVAEDebiaser.TensorControl",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "FluxVAEDebiaser") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                if (node._fluxVaeInit) return;
                node._fluxVaeInit = true;
                node.combineBlockWidgets();
                node.setupPresetWidget();
                if (node.size[0] < 540) {
                    node.size[0] = 540;
                    node.setDirtyCanvas(true);
                }
            }, 50);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                for (const blockName of ALL_BLOCKS) {
                    const strWidget = node.widgets.find(w => w.name === blockName + "_str");
                    if (strWidget) {
                        let val = parseFloat(strWidget.value);
                        if (isNaN(val)) strWidget.value = 1.0;
                    }
                }
                node.setDirtyCanvas(true);
            }, 150);
        };

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(output) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            if (output && output.analysis_json && output.analysis_json[0]) {
                try {
                    const jsonStr = output.analysis_json[0];
                    if (jsonStr && jsonStr.length > 2) {
                        this._analysisData = JSON.parse(jsonStr);
                        this.setDirtyCanvas(true);
                    }
                } catch (e) {}
            }
        };

        nodeType.prototype.combineBlockWidgets = function() {
            const widgetPairs = [];
            const strWidgetNames = new Set();
            for (const widget of this.widgets) {
                if (widget.name.endsWith('_str')) strWidgetNames.add(widget.name);
            }
            for (const widget of this.widgets) {
                const strName = widget.name + '_str';
                if (strWidgetNames.has(strName)) {
                    const strWidget = this.widgets.find(w => w.name === strName);
                    if (strWidget) widgetPairs.push({ toggle: widget, strength: strWidget, name: widget.name });
                }
            }
            for (const pair of widgetPairs) this.createCombinedWidget(pair);
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true);
        };

        nodeType.prototype.createCombinedWidget = function(pair) {
            const { toggle, strength, name } = pair;

            toggle.draw = function(ctx, node, widgetWidth, y, widgetHeight) {
                const margin = 10;
                const checkboxSize = 14;
                const labelWidth = 165;
                const valueWidth = 38;
                const gap = 6;
                const sliderWidth = widgetWidth - margin - checkboxSize - gap - labelWidth - gap - valueWidth - margin - gap;
                const checkboxX = margin;
                const labelX = checkboxX + checkboxSize + gap;
                const sliderX = labelX + labelWidth + gap;

                const enabled = Boolean(toggle.value);
                let strengthVal = parseFloat(strength.value);
                if (isNaN(strengthVal)) strengthVal = 1.0;

                let impactScore = null;
                if (node._analysisData && node._analysisData.blocks) {
                    const blockData = node._analysisData.blocks[name];
                    if (blockData && typeof blockData.score === 'number') impactScore = blockData.score;
                }

                let checkboxColor = getSubColor(name);
                if (impactScore !== null) checkboxColor = getImpactColor(impactScore);

                ctx.fillStyle = getSubBg(name, enabled);
                ctx.fillRect(0, y, widgetWidth, widgetHeight);

                // Checkbox
                ctx.strokeStyle = enabled ? checkboxColor : "#555";
                ctx.lineWidth = 1.5;
                ctx.strokeRect(checkboxX, y + (widgetHeight - checkboxSize) / 2, checkboxSize, checkboxSize);
                ctx.globalAlpha = enabled ? 1.0 : 0.35;
                ctx.fillStyle = checkboxColor;
                ctx.fillRect(checkboxX + 2, y + (widgetHeight - checkboxSize) / 2 + 2, checkboxSize - 4, checkboxSize - 4);
                ctx.globalAlpha = 1.0;

                // Label
                ctx.fillStyle = enabled ? "#ddd" : "#666";
                ctx.font = "11px Arial";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                let displayLabel = name;
                const maxLabelWidth = labelWidth - 4;
                if (ctx.measureText(displayLabel).width > maxLabelWidth) {
                    while (ctx.measureText(displayLabel + "…").width > maxLabelWidth && displayLabel.length > 3) {
                        displayLabel = displayLabel.slice(0, -1);
                    }
                    displayLabel += "…";
                }
                ctx.fillText(displayLabel, labelX, y + widgetHeight / 2);

                // Slider track
                const trackY = y + widgetHeight / 2;
                const trackHeight = 4;
                const min = -2.0, max = 2.0, range = max - min;
                const normalizedStrength = (strengthVal - min) / range;
                const zeroPos = (0 - min) / range;

                ctx.fillStyle = "#333";
                ctx.beginPath();
                ctx.roundRect(sliderX, trackY - trackHeight / 2, sliderWidth, trackHeight, 2);
                ctx.fill();

                // Filled portion
                ctx.fillStyle = enabled ? (strengthVal >= 0 ? "#5599ff" : "#ff6655") : "#444";
                const zeroX = sliderX + zeroPos * sliderWidth;
                const strengthX = sliderX + normalizedStrength * sliderWidth;
                ctx.beginPath();
                ctx.roundRect(Math.min(zeroX, strengthX), trackY - trackHeight / 2, Math.abs(strengthX - zeroX), trackHeight, 2);
                ctx.fill();

                // Thumb
                ctx.fillStyle = enabled ? "#fff" : "#666";
                ctx.beginPath();
                ctx.arc(sliderX + normalizedStrength * sliderWidth, trackY, 5, 0, Math.PI * 2);
                ctx.fill();

                // Value text
                ctx.fillStyle = enabled ? "#ddd" : "#555";
                ctx.textAlign = "right";
                ctx.font = "10px Arial";
                ctx.fillText(strengthVal.toFixed(2), widgetWidth - margin, y + widgetHeight / 2);
            };

            toggle.sliderInfo = {
                margin: 10, checkboxSize: 14, labelWidth: 165, valueWidth: 38, gap: 6,
                min: -5.0, max: 5.0, step: 0.01,
                getLayout: function(widgetWidth) {
                    const sliderWidth = widgetWidth - this.margin - this.checkboxSize - this.gap - this.labelWidth - this.gap - this.valueWidth - this.margin - this.gap;
                    const checkboxX = this.margin;
                    const sliderX = checkboxX + this.checkboxSize + this.gap + this.labelWidth + this.gap;
                    return { sliderWidth, checkboxX, sliderX };
                }
            };

            const originalMouse = toggle.mouse?.bind(toggle);
            toggle.mouse = function(event, pos, node) {
                const widgetWidth = node.size[0];
                const layout = toggle.sliderInfo.getLayout(widgetWidth);
                const valueX = widgetWidth - toggle.sliderInfo.margin - toggle.sliderInfo.valueWidth;
                const localX = pos[0];

                // Click on the numeric value column → prompt for direct entry.
                // Without this, clicks fell through to the boolean toggle handler
                // and ended up disabling the row instead of editing the value.
                if (event.type === "pointerdown" && localX >= valueX - 4) {
                    const current = parseFloat(strength.value);
                    const seed = isNaN(current) ? "1.0" : current.toFixed(2);
                    const input = window.prompt(
                        `Set strength (${toggle.sliderInfo.min.toFixed(1)} to ${toggle.sliderInfo.max.toFixed(1)}):`,
                        seed
                    );
                    if (input !== null && input !== "") {
                        const parsed = parseFloat(input);
                        if (!isNaN(parsed)) {
                            let v = Math.max(toggle.sliderInfo.min, Math.min(toggle.sliderInfo.max, parsed));
                            v = Math.round(v / toggle.sliderInfo.step) * toggle.sliderInfo.step;
                            strength.value = v;
                            node.setDirtyCanvas(true);
                        }
                    }
                    return true;
                }

                // Slider drag — RELATIVE / anchored (matches the flux_klein debiaser, #42).
                // Grab at the current value instead of jumping to the cursor; adjust from
                // the grab point so small drags give fine control and you never have to
                // drag to the track edge to reach min/max.
                if (event.type === "pointerdown" && localX >= layout.sliderX - 5 && localX <= layout.sliderX + layout.sliderWidth + 5) {
                    toggle._strDrag = { x: localX, v: parseFloat(strength.value) || 0 };
                    return true;
                }
                if (event.type === "pointermove" && toggle._strDrag) {
                    if (event.buttons === 0) {
                        toggle._strDrag = null;
                    } else {
                        let newStr = toggle._strDrag.v + ((localX - toggle._strDrag.x) / layout.sliderWidth) * (toggle.sliderInfo.max - toggle.sliderInfo.min);
                        newStr = Math.round(newStr / toggle.sliderInfo.step) * toggle.sliderInfo.step;
                        newStr = Math.max(toggle.sliderInfo.min, Math.min(toggle.sliderInfo.max, newStr));
                        strength.value = newStr;
                        node.setDirtyCanvas(true);
                        return true;
                    }
                }
                if (event.type === "pointerup" && toggle._strDrag) {
                    toggle._strDrag = null;
                    return true;
                }
                if (originalMouse) return originalMouse(event, pos, node);
                return false;
            };

            strength.draw = function() {};
            strength.computeSize = function() { return [0, -4]; };
        };

        nodeType.prototype.setupPresetWidget = function() {
            const node = this;
            const presetWidget = this.widgets.find(w => w.name === "preset");
            if (!presetWidget) return;
            const presetNames = Object.keys(DEEP_PRESETS);
            presetWidget.options = presetWidget.options || {};
            presetWidget.options.values = presetNames;
            if (!presetNames.includes(presetWidget.value)) presetWidget.value = "Default";
            const origCallback = presetWidget.callback;
            presetWidget.callback = function(value) {
                node.applyPreset(value);
                if (origCallback) origCallback.call(this, value);
            };
            presetWidget.serializeValue = function() { return "Custom"; };
            this.presetWidget = presetWidget;
            this.setSize(this.computeSize());
            this.setDirtyCanvas(true);
        };

        nodeType.prototype.applyPreset = function(presetName) {
            const preset = DEEP_PRESETS[presetName];
            if (!preset) return;
            const enabledBlocks = preset.enabled === "ALL" ? ALL_BLOCKS : preset.enabled;
            const enabledSet = new Set(enabledBlocks);
            const baseStrength = preset.strength;
            const overrides = preset.overrides || {};
            for (const blockName of ALL_BLOCKS) {
                const toggleWidget = this.widgets.find(w => w.name === blockName);
                const strWidget = this.widgets.find(w => w.name === blockName + "_str");
                if (toggleWidget) toggleWidget.value = enabledSet.has(blockName);
                if (strWidget) strWidget.value = overrides.hasOwnProperty(blockName) ? overrides[blockName] : baseStrength;
            }
            this.setDirtyCanvas(true);
        };
    }
});
