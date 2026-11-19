const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const extensions = [];
let now = 0;
let nextTimer = 0;
const timers = new Map();
const context = {
    app: { registerExtension: extension => extensions.push(extension) },
    console,
    setTimeout(callback, delay) {
        const id = ++nextTimer;
        timers.set(id, { callback, at: now + delay });
        return id;
    },
    clearTimeout(id) { timers.delete(id); },
};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../web/js/dynamic_inputs.js"), "utf8")
    .replace(/^import .*;\r?\n/, ""), context);

const expectedMarks = ["", "\u2605", "\u25a0", "\u25b2", "\u2191", "\u2193", "\u00d7", ""];
const expectedColors = ["#777", "#f4cf71", "#79b8ff", "#c79bff", "#73d99f", "#ffad66", "#ff7b72", "#777"];

function draw(toggle, node, y = 37) {
    const texts = [];
    texts.borders = [];
    texts.positions = [];
    const ctx = {
        fillRect() {}, beginPath() {}, roundRect() {}, fill() {}, arc() {},
        strokeRect(x, y, width, height) { texts.borders.push({ x, y, width, height }); },
        measureText(text) { return { width: text.length * 5 }; },
        fillText(text, x) { texts.push(text); texts.positions.push({ text, x, color: this.fillStyle }); },
    };
    toggle.draw(ctx, node, node.size[0], y, 20);
    return texts;
}

function advanceTime(milliseconds) {
    const end = now + milliseconds;
    for (;;) {
        const next = [...timers].filter(([, timer]) => timer.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        const [id, timer] = next;
        now = timer.at;
        timers.delete(id);
        timer.callback();
    }
    now = end;
}

function nativePress(toggle, node, position) {
    const canvas = { graph_mouse: [position[0] + node.pos[0], position[1] + node.pos[1]] };
    const pointer = {};
    if (!toggle.onPointerDown(pointer, node, canvas)) {
        // This is the BooleanWidget default that marker clicks must bypass.
        pointer.onClick = () => { toggle.value = !toggle.value; };
    }
    return {
        pointer,
        release(double = false) {
            (double ? pointer.onDoubleClick : pointer.onClick)?.();
            pointer.finally?.();
        },
    };
}

function nativeClick(toggle, node, position, double = false) {
    nativePress(toggle, node, position).release(double);
}

(async () => {
    const extension = extensions.find(item => item.name === "SelectiveLoRA.CombinedWidgets");
    const names = vm.runInNewContext("Object.keys(SELECTIVE_LOADER_PRESETS)", context);
    for (const name of names) {
        class Node {
            constructor() {
                this.widgets = [];
                this.pos = [100, 200];
                this.size = [500, 400];
                this.properties = { unrelated: "keep" };
                this.changes = [];
                this.graph = {
                    beforeChange: () => this.changes.push(JSON.stringify(this.properties)),
                    afterChange() {},
                };
            }
            setDirtyCanvas() {}
        }
        await extension.beforeRegisterNodeDef(Node, { name }, context.app);
        const node = new Node();
        const blocks = vm.runInNewContext(`SELECTIVE_LOADER_PRESETS[${JSON.stringify(name)}].blocks`, context);
        const block = blocks[0];
        const other = blocks[blocks.length - 1];
        const toggle = { name: block, value: false };
        const strength = { name: `${block}_str`, value: -0.55, options: { min: -5, max: 5 } };
        const otherToggle = { name: other, value: true };
        const otherStrength = { name: `${other}_str`, value: 0.2, options: { min: -5, max: 5 } };
        node.widgets.push(toggle, strength, otherToggle, otherStrength, { name: "block_weights_string", value: "%default=1" });
        node.createCombinedWidget({ toggle, strength, name: block });
        node.createCombinedWidget({ toggle: otherToggle, strength: otherStrength, name: other });
        const widgetValues = JSON.stringify(node.widgets.map(widget => widget.value));
        const widgetCount = node.widgets.length;
        const initialDrawing = draw(toggle, node);
        draw(otherToggle, node, 65);
        const layout = toggle.sliderInfo.getLayout(node.size[0]);
        const marker = [layout.markerX + toggle.sliderInfo.markerWidth / 2, 47];
        assert.equal(layout.labelX, 30);
        assert.equal(layout.sliderX, 131);
        assert.equal(layout.sliderWidth, 315);
        assert.ok(layout.markerX > layout.labelX);
        assert.ok(layout.markerX + toggle.sliderInfo.markerWidth < layout.sliderX);
        assert.ok(layout.labelX + toggle.sliderInfo.labelWidth < layout.sliderX);
        assert.equal(initialDrawing.borders.length, 1, "Only the existing checkbox has a border");
        assert.equal(initialDrawing.borders[0].width, toggle.sliderInfo.checkboxSize);
        const dot = initialDrawing.positions.find(item => item.text === "\u00b7");
        assert.equal(dot.color, expectedColors[0]);
        const label = initialDrawing.positions.find(item => item.x === layout.labelX);
        assert.ok(label.x + label.text.length * 5 < dot.x);
        assert.ok(dot.x < layout.sliderX);

        for (let i = 1; i < expectedMarks.length; i++) {
            nativeClick(toggle, node, marker, i % 2 === 0);
            const mark = expectedMarks[i];
            assert.equal(node.properties.lora_layer_marks[block] || "", mark, name);
            const drawing = draw(toggle, node);
            assert.ok(drawing.includes(mark || "\u00b7"));
            assert.equal(drawing.positions.find(item => item.text === (mark || "\u00b7")).color, expectedColors[i]);
            assert.equal(JSON.stringify(node.widgets.map(widget => widget.value)), widgetValues);
            assert.equal(node.widgets.length, widgetCount);
            assert.equal(node.properties.unrelated, "keep");
        }
        assert.equal(node.changes.length, 7);
        assert.deepEqual(JSON.parse(node.changes[0]), { unrelated: "keep" });
        assert.equal(Object.keys(node.properties.lora_layer_marks).length, 0);

        nativeClick(toggle, node, marker);
        nativeClick(otherToggle, node, [marker[0], 75]);
        const savedWorkflowNode = JSON.parse(JSON.stringify({ properties: node.properties }));
        const restored = new Node();
        restored.properties = savedWorkflowNode.properties;
        assert.ok(draw(toggle, restored).includes("\u2605"));
        const clone = new Node();
        clone.properties = { ...node.properties };
        nativeClick(toggle, clone, marker);
        assert.equal(clone.properties.lora_layer_marks[block], "\u25a0");
        assert.equal(node.properties.lora_layer_marks[block], "\u2605");
        assert.equal(node.properties.lora_layer_marks[other], "\u2605");
        node.applyPreset(name, "All Off");
        assert.equal(node.properties.lora_layer_marks[block], "\u2605");

        // Adjacent row, label, and slider clicks must not be captured by the marker.
        for (const pos of [[marker[0], 20], [layout.labelX + 4, 47], [layout.sliderX + 10, 47]]) {
            assert.equal(Boolean(toggle.onPointerDown({}, node, { graph_mouse: [pos[0] + 100, pos[1] + 200] })), false);
        }
        toggle.mouse({ type: "pointerdown" }, marker, node);
        toggle.mouse({ type: "pointerup" }, marker, node);
        assert.equal(node.properties.lora_layer_marks[block], "\u25a0");
        const currentMark = node.properties.lora_layer_marks[block];
        toggle.mouse({ type: "pointerdown" }, marker, node);
        toggle.mouse({ type: "pointermove" }, [layout.sliderX + 10, 47], node);
        toggle.mouse({ type: "pointerup" }, marker, node);
        assert.equal(node.properties.lora_layer_marks[block], currentMark);

        // A slider drag across the marker still clamps to the track endpoint.
        const pos = value => [layout.sliderX + (value + 5) / 10 * layout.sliderWidth, 47];
        toggle.mouse({ type: "pointerdown" }, pos(0.55), node);
        toggle.mouse({ type: "pointermove" }, pos(0.6), node);
        assert.ok(Math.abs(strength.value - 0.6) < 1e-8);
        toggle.mouse({ type: "pointermove" }, marker, node);
        assert.equal(strength.value, -5);
        toggle.mouse({ type: "pointermove" }, pos(5), node);
        assert.equal(strength.value, 5);
        toggle.mouse({ type: "pointerup" }, pos(5), node);
        assert.equal(node.properties.lora_layer_marks[block], currentMark);

        const beforeHold = JSON.stringify(node.widgets.map(widget => widget.value));
        const otherMark = node.properties.lora_layer_marks[other];
        const held = nativePress(toggle, node, marker);
        advanceTime(1499);
        assert.equal(node.properties.lora_layer_marks[block], currentMark);
        advanceTime(1);
        assert.equal(node.properties.lora_layer_marks[block], undefined);
        assert.ok(draw(toggle, node).includes("\u00b7"));
        held.release(true);
        assert.equal(node.properties.lora_layer_marks[block], undefined, "Release after a hold must not cycle");
        assert.equal(node.properties.lora_layer_marks[other], otherMark);
        assert.equal(JSON.stringify(node.widgets.map(widget => widget.value)), beforeHold);
        const changesBeforeBlankHold = node.changes.length;
        const blankHold = nativePress(toggle, node, marker);
        advanceTime(2000);
        blankHold.release();
        assert.equal(node.properties.lora_layer_marks[block], undefined);
        assert.equal(node.changes.length, changesBeforeBlankHold);

        nativeClick(toggle, node, marker);
        advanceTime(2000);
        assert.equal(node.properties.lora_layer_marks[block], "\u2605", "Short click timers must be canceled");
        const canceled = nativePress(toggle, node, marker);
        advanceTime(500);
        canceled.pointer.finally();
        advanceTime(2000);
        assert.equal(node.properties.lora_layer_marks[block], "\u2605");
        const dragged = nativePress(toggle, node, marker);
        advanceTime(500);
        dragged.pointer.onDragStart();
        advanceTime(2000);
        dragged.pointer.finally();
        assert.equal(node.properties.lora_layer_marks[block], "\u2605");

        toggle.mouse({ type: "pointerdown" }, marker, node);
        advanceTime(1500);
        assert.equal(node.properties.lora_layer_marks[block], undefined);
        toggle.mouse({ type: "pointerup" }, marker, node);
        assert.equal(node.properties.lora_layer_marks[block], undefined);
        toggle.mouse({ type: "pointerdown" }, marker, node);
        toggle.mouse({ type: "pointerup" }, marker, node);
        assert.equal(node.properties.lora_layer_marks[block], "\u2605");
        toggle.mouse({ type: "pointerdown" }, marker, node);
        advanceTime(500);
        toggle.mouse({ type: "pointercancel" }, marker, node);
        advanceTime(2000);
        assert.equal(node.properties.lora_layer_marks[block], "\u2605");
        assert.equal(timers.size, 0);
    }
    console.log(`Layer markers passed for ${names.length} selective loaders, combined nodes, and model layer editors`);
    console.log("Verified symbol cycle, hold-to-clear, cancellation, workflow persistence, clone independence, undo snapshots, and slider isolation");
})().catch(error => { console.error(error); process.exitCode = 1; });
