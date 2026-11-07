// Exercise the existing widget hooks with both new MOSS node definitions.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const extensions = [];
const timers = [];
const app = { registerExtension: (extension) => extensions.push(extension) };
const context = { app, console, setTimeout: (fn) => timers.push(fn) };
const script = fs.readFileSync(path.join(__dirname, "../web/js/dynamic_inputs.js"), "utf8")
    .replace(/^import .*;\r?\n/, "");
vm.runInNewContext(script, context);
const analyzerPayloads = process.argv.includes("--analysis-payloads")
    ? JSON.parse(fs.readFileSync(0, "utf8")) : null;

function checkboxColor(node, name) {
    const toggle = node.widgets.find((widget) => widget.name === name);
    const strength = node.widgets.find((widget) => widget.name === `${name}_str`);
    node.createCombinedWidget({ toggle, strength, name });
    const colors = [];
    const ctx = {
        fillRect(x, y, w, h) { if (w === 10 && h === 10) colors.push(this.fillStyle); },
        strokeRect() {}, fillText() {}, beginPath() {}, roundRect() {}, fill() {}, arc() {},
        measureText(value) { return { width: value.length * 5 }; }
    };
    toggle.draw(ctx, node, 500, 0, 20);
    return colors[0];
}

(async () => {
    const extension = extensions.find((item) => item.name === "SelectiveLoRA.CombinedWidgets");
    for (const name of ["MOSSImageSelectiveLoRALoader", "MOSSImageAnalyzerSelectiveLoaderV2"]) {
        class Node {
            constructor() { this.widgets = []; this.size = [500, 400]; }
            setDirtyCanvas() {}
        }
        await extension.beforeRegisterNodeDef(Node, { name }, app);
        const node = new Node();
        const events = {};
        const text = {
            name: "block_weights_string", value: "",
            inputEl: { addEventListener: (event, fn) => events[event] = fn }
        };
        node.widgets.push(text);
        for (const block of [...Array.from({length: 42}, (_, i) => `layer_${i}`), "other_weights"]) {
            node.widgets.push({ name: block, value: true });
            node.widgets.push({ name: `${block}_str`, value: 1, options: { min: -5, max: 5, step: 0.5 } });
        }
        const widget = (name) => node.widgets.find((item) => item.name === name);
        node.setupBidirectionalSync(name);
        while (timers.length) timers.shift()();
        const values = Array(43).fill("1.00");
        values[1] = "-0.55";
        values[2] = "0.00";
        values[42] = "0.20";
        text.value = values.join(", ");
        events.input();
        assert.equal(widget("layer_1_str").value, -0.55);
        assert.equal(widget("layer_2").value, false);
        assert.equal(widget("other_weights_str").value, 0.2);
        widget("layer_1_str").value = 1.3;
        widget("layer_1_str").callback(1.3);
        assert.equal(text.value.split(", ")[1], "1.30");
        text.value = Array(42).fill("0.50").join(", ");
        events.input();
        assert.equal(widget("layer_41_str").value, 0.5);
        assert.equal(widget("other_weights_str").value, 1);
        text.value = "%default=0, layer14-27=1.3, layer_41=-0.5, other=0.2";
        events.input();
        assert.equal(widget("layer_13").value, false);
        assert.equal(widget("layer_14_str").value, 1.3);
        assert.equal(widget("layer_41_str").value, -0.5);
        assert.equal(widget("other_weights_str").value, 0.2);

        const toggle = widget("layer_1");
        const strength = widget("layer_1_str");
        node.createCombinedWidget({ toggle, strength, name: "layer_1" });
        toggle._lastDrawY = 0;
        toggle._lastDrawHeight = 20;
        const layout = toggle.sliderInfo.getLayout(500);
        const pos = (value) => [layout.sliderX + (value + 5) / 10 * layout.sliderWidth, 10];
        toggle.mouse({ type: "pointerdown" }, pos(0.55), node);
        assert.ok(Math.abs(strength.value - 0.55) < 1e-8);
        toggle.mouse({ type: "pointermove" }, pos(0.6), node);
        assert.ok(Math.abs(strength.value - 0.6) < 1e-8);
        toggle.mouse({ type: "pointermove" }, pos(-5), node);
        assert.equal(strength.value, -5);
        toggle.mouse({ type: "pointermove" }, pos(5), node);
        assert.equal(strength.value, 5);
        toggle.mouse({ type: "pointerup" }, pos(5), node);
        node.applyPreset(name, "Late Only (28-41)");
        assert.equal(widget("layer_27").value, false);
        assert.equal(widget("layer_28").value, true);
        assert.equal(widget("other_weights").value, true);
        const payloads = analyzerPayloads || {
            LoRALoaderWithAnalysis: { analysis_json: [JSON.stringify({ blocks: { layer_1: { score: 75 } } })] },
            LoRALoaderWithAnalysisV2: { analysis_json: [JSON.stringify({ blocks: { layer_1: { score: 35 } } })] },
        };
        const storeExtension = extensions.find((item) => item.name === "LoRAAnalyzer.StoreAnalysis");
        for (const [analyzerName, payload] of Object.entries(payloads)) {
            class Analyzer {}
            await storeExtension.beforeRegisterNodeDef(Analyzer, { name: analyzerName }, app);
            const source = new Analyzer();
            source.onExecuted(payload);
            node.inputs = [{ name: "analysis_json", link: 1 }];
            node.graph = { links: { 1: { origin_id: 100 } }, getNodeById: () => source };
            node._analysisData = { blocks: { layer_1: { score: 0 } } };
            widget("layer_1").value = true;
            const score = JSON.parse(payload.analysis_json[0]).blocks.layer_1.score;
            const expected = vm.runInNewContext(`getImpactColor(${score})`, context);
            assert.equal(checkboxColor(node, "layer_1"), expected);
            source.onExecuted({ analysis_json: ['{"blocks":{"layer_1":{"score":75}}}'] });
            assert.equal(checkboxColor(node, "layer_1"), "#ff9900");
            source.onExecuted({ analysis_json: ['{"blocks":{"layer_1":{"score":35}}}'] });
            assert.equal(checkboxColor(node, "layer_1"), "#00cccc");
        }
        node.inputs = [];
        node.onExecuted({ analysis_json: ['{"blocks":{"layer_1":{"score":75}}}'] });
        assert.equal(checkboxColor(node, "layer_1"), "#ff9900");
        node.graph = { links: { 1: { origin_id: 100 } }, getNodeById: () => ({ _analysisData: { blocks: { layer_1: { score: 35 } } } }) };
        node.inputs = [{ name: "analysis_json", link: 1 }];
        assert.equal(checkboxColor(node, "layer_1"), "#00cccc");
        console.log(`${name}: text sync, presets, click-to-jump, 0.05 steps and full range passed`);
        console.log(`${name}: connected analyzer payloads and rendered layer colors passed`);
    }
    const sharedNodeNames = vm.runInNewContext("Object.keys(SELECTIVE_LOADER_PRESETS)", context)
        .filter(name => !name.includes("ModelLayerEditor"));
    for (const name of sharedNodeNames) {
        class Node {
            constructor() { this.widgets = []; this.size = [500, 400]; }
            setDirtyCanvas() {}
        }
        await extension.beforeRegisterNodeDef(Node, { name }, app);
        const node = new Node();
        const block = vm.runInNewContext(`SELECTIVE_LOADER_PRESETS[${JSON.stringify(name)}].blocks[0]`, context);
        for (const name of [block, "other_weights"]) {
            node.widgets.push({ name, value: true });
            node.widgets.push({ name: `${name}_str`, value: 1, options: { min: -5, max: 5 } });
        }
        const payload = { analysis_json: [JSON.stringify({ blocks: { [block]: { score: 75 }, other: { score: 35 } } })] };
        node.onExecuted(payload);
        assert.equal(checkboxColor(node, block), "#ff9900", name);
        assert.equal(checkboxColor(node, "other_weights"), "#00cccc", name);
        node.graph = { links: { 1: { origin_id: 100 } }, getNodeById: () => ({ _lastAnalysisData: { blocks: { [block]: { score: 35 } } } }) };
        node.inputs = [{ name: "analysis_json", link: 1 }];
        assert.equal(checkboxColor(node, block), "#00cccc", name);
    }
    console.log(`Shared layer-color display passed for ${sharedNodeNames.length} selective and combined node types`);
})().catch((error) => { console.error(error); process.exitCode = 1; });
