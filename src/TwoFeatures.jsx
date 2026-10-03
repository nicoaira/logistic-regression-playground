import React, { useMemo, useState } from 'react';
import { Play, Pause, RotateCcw, Target } from 'lucide-react';
import { STUDENTS, BEST_2D, X1_RANGE, X2_RANGE } from './data';
import { sigmoid, logit, cost, range, lineInBox, halfPlaneInBox } from './logistic';
import { Plot, COLORS, Slider, Metric, Toggle, useTrainer, baseLayout, PLOT_CONFIG, copy } from './ui';

const XS = STUDENTS.map((s) => [s.x1, s.x2]);
const YS = STUDENTS.map((s) => s.y);
const GX = range(X1_RANGE[0], X1_RANGE[1], 41);
const GY = range(X2_RANGE[0], X2_RANGE[1], 29);
const MX = range(X1_RANGE[0], X1_RANGE[1], 81);
const MY = range(X2_RANGE[0], X2_RANGE[1], 57);
// Students are drawn this far above their label (0 or 1) in 3D: the sheet sits at almost exactly 0 or 1
// there, and markers at the same height flicker in and out of view (z-fighting).
const LIFT = 0.015;

export default function TwoFeatures({ pTheme }) {
    const [w1, setW1] = useState(BEST_2D.w1);
    const [w2, setW2] = useState(BEST_2D.w2);
    const [b, setB] = useState(BEST_2D.b);
    const [c, setC] = useState(0.5);
    const [showErrors, setShowErrors] = useState(true);
    const [showMap, setShowMap] = useState(false);
    const [showGhost, setShowGhost] = useState(true);

    const trainer = useTrainer({
        getParams: () => ({ w: [w1, w2], b }),
        setParams: (w, bNew) => { setW1(w[0]); setW2(w[1]); setB(bNew); },
        xs: XS, ys: YS, alpha: 0.05, stepsPerFrame: 2000, maxSteps: 600000,
    });
    const stopAnd = (fn) => (v) => { trainer.setPlaying(false); fn(v); };

    const t = logit(c);

    const m = useMemo(() => {
        const probs = STUDENTS.map((s) => sigmoid(w1 * s.x1 + w2 * s.x2 + b));
        const wrong = STUDENTS.map((s, i) => (probs[i] >= c ? 1 : 0) !== s.y);
        return { probs, wrong, nWrong: wrong.filter(Boolean).length, J: cost(probs, YS) };
    }, [w1, w2, b, c]);

    const surfaceZ = useMemo(() => GY.map((y) => GX.map((x) => sigmoid(w1 * x + w2 * y + b))), [w1, w2, b]);
    const mapZ = useMemo(() => MY.map((y) => MX.map((x) => sigmoid(w1 * x + w2 * y + b))), [w1, w2, b]);

    // Decision boundary for the current cut-off: w1·x1 + w2·x2 + b = logit(c), clipped to the plot box.
    const seg = lineInBox(w1, w2, t - b, X1_RANGE, X2_RANGE);
    const ghost = lineInBox(w1, w2, -b, X1_RANGE, X2_RANGE); // the c = 0.5 boundary, for reference
    const hasBoundary = seg.length === 2;
    const ratio = Math.abs(w2) > 1e-9 ? w1 / w2 : null;
    // Angle between the boundary and the hours-studied axis (90° when w2 = 0: a vertical boundary).
    const angle = ratio === null ? 90 : (Math.abs(Math.atan(ratio)) * 180) / Math.PI;

    const fails = STUDENTS.map((s, i) => ({ ...s, i })).filter((s) => s.y === 0);
    const passes = STUDENTS.map((s, i) => ({ ...s, i })).filter((s) => s.y === 1);
    const errs = STUDENTS.map((s, i) => ({ ...s, i })).filter((s) => m.wrong[s.i]);

    // ── 3D: the probability surface, the threshold plane and where they meet ──
    const plot3d = [
        {
            type: 'surface', x: copy(GX), y: copy(GY), z: surfaceZ, cmin: 0, cmax: 1, showscale: false, opacity: 0.85,
            colorscale: [[0, COLORS.fail], [0.5, pTheme.midColor], [1, COLORS.pass]],
            hovertemplate: 'studied %{x:.1f} h<br>slept %{y:.1f} h<br>f = %{z:.3f}<extra></extra>',
            lighting: { ambient: 0.75, diffuse: 0.5, specular: 0.1 },
        },
        {
            type: 'surface', x: copy(X1_RANGE), y: copy(X2_RANGE), z: [[c, c], [c, c]], showscale: false, opacity: 0.38,
            colorscale: [[0, COLORS.threshold], [1, COLORS.threshold]], hoverinfo: 'skip',
        },
        {
            type: 'scatter3d', mode: 'markers', x: fails.map((s) => s.x1), y: fails.map((s) => s.x2), z: fails.map(() => LIFT),
            marker: { color: COLORS.fail, size: 5, symbol: 'circle', line: { color: 'white', width: 1 } },
            hovertemplate: 'fail · %{x} h studied, %{y} h slept<extra></extra>',
        },
        {
            type: 'scatter3d', mode: 'markers', x: passes.map((s) => s.x1), y: passes.map((s) => s.x2), z: passes.map(() => 1 + LIFT),
            marker: { color: COLORS.pass, size: 4, symbol: 'x' },
            hovertemplate: 'pass · %{x} h studied, %{y} h slept<extra></extra>',
        },
    ];
    if (showErrors) {
        plot3d.push({
            type: 'scatter3d', mode: 'markers', x: errs.map((s) => s.x1), y: errs.map((s) => s.x2), z: errs.map((s) => s.y + LIFT),
            marker: { color: COLORS.error, size: 8, symbol: 'circle-open', line: { width: 2 } }, hoverinfo: 'skip',
        });
    }
    if (hasBoundary) {
        plot3d.push({
            type: 'scatter3d', mode: 'lines', x: seg.map((p) => p[0]), y: seg.map((p) => p[1]), z: [c, c],
            line: { color: pTheme.fontColor, width: 8 }, hoverinfo: 'skip',
        });
        plot3d.push({
            type: 'scatter3d', mode: 'lines', x: seg.map((p) => p[0]), y: seg.map((p) => p[1]), z: [0, 0],
            line: { color: COLORS.threshold, width: 7 }, hoverinfo: 'skip',
        });
    }
    const axis3d = (title, rng) => ({
        title: { text: title }, range: rng, gridcolor: pTheme.gridColor, zerolinecolor: pTheme.gridColor,
        showbackground: true, backgroundcolor: pTheme.paneColor,
    });

    // ── 2D: the input plane seen from above ──
    const plot2d = [];
    if (showMap) {
        plot2d.push({
            type: 'contour', x: copy(MX), y: copy(MY), z: mapZ, zmin: 0, zmax: 1, showscale: false, hoverinfo: 'skip',
            colorscale: [[0, 'rgba(59,130,246,0.45)'], [0.5, 'rgba(0,0,0,0)'], [1, 'rgba(249,115,22,0.45)']],
            contours: { start: 0.1, end: 0.9, size: 0.1, coloring: 'heatmap', showlabels: true, labelfont: { size: 10, color: pTheme.fontColor } },
            line: { color: 'rgba(148,163,184,0.6)', width: 0.8 },
        });
    } else {
        const passPoly = halfPlaneInBox(w1, w2, t - b, 1, X1_RANGE, X2_RANGE);
        const failPoly = halfPlaneInBox(w1, w2, t - b, -1, X1_RANGE, X2_RANGE);
        for (const [poly, col] of [[failPoly, COLORS.failRegion], [passPoly, COLORS.passRegion]]) {
            if (poly.length > 2) {
                plot2d.push({ type: 'scatter', mode: 'lines', x: poly.map((p) => p[0]), y: poly.map((p) => p[1]), fill: 'toself', fillcolor: col, line: { width: 0 }, hoverinfo: 'skip' });
            }
        }
    }
    if (showGhost && Math.abs(c - 0.5) > 0.005 && ghost.length === 2) {
        plot2d.push({ type: 'scatter', mode: 'lines', x: ghost.map((p) => p[0]), y: ghost.map((p) => p[1]), line: { color: COLORS.muted, width: 2, dash: 'dash' }, hoverinfo: 'skip' });
    }
    if (hasBoundary) {
        plot2d.push({ type: 'scatter', mode: 'lines', x: seg.map((p) => p[0]), y: seg.map((p) => p[1]), line: { color: COLORS.threshold, width: 4 }, hoverinfo: 'skip' });
    }
    plot2d.push(
        { type: 'scatter', mode: 'markers', x: fails.map((s) => s.x1), y: fails.map((s) => s.x2), marker: { color: COLORS.fail, size: 10, line: { color: 'white', width: 1 } }, hovertemplate: 'fail · %{x} h, %{y} h<extra></extra>' },
        { type: 'scatter', mode: 'markers', x: passes.map((s) => s.x1), y: passes.map((s) => s.x2), marker: { color: COLORS.pass, size: 11, symbol: 'x', line: { color: 'white', width: 1 } }, hovertemplate: 'pass · %{x} h, %{y} h<extra></extra>' },
    );
    if (showErrors) {
        plot2d.push({ type: 'scatter', mode: 'markers', x: errs.map((s) => s.x1), y: errs.map((s) => s.x2), marker: { color: COLORS.error, size: 20, symbol: 'circle-open', line: { width: 2 } }, hoverinfo: 'skip' });
    }

    const tiltText = ratio === null ? 'tilt: w₂ = 0 (vertical boundary)' : `tilt: 1 h study = ${ratio.toFixed(2)} h sleep (${angle.toFixed(1)}°)`;

    return (
        <>
            <div className="plots-grid glass-panel">
                <div className="panel">
                    <div className="panel-header">
                        <h3>3D: the sheet f(x₁, x₂) and the threshold plane</h3>
                        <Toggle checked={showErrors} onChange={setShowErrors} color={COLORS.error}>Show mistakes</Toggle>
                    </div>
                    <Plot
                        data={plot3d}
                        layout={{
                            ...baseLayout(pTheme),
                            margin: { t: 0, r: 0, l: 0, b: 0 },
                            scene: {
                                xaxis: axis3d('hours studied (x₁)', copy(X1_RANGE)),
                                yaxis: axis3d('hours slept (x₂)', copy(X2_RANGE)),
                                zaxis: { ...axis3d('probability f', [0, 1 + 2 * LIFT]), tickvals: [0, 0.5, 1] },
                                camera: { eye: { x: 1.3, y: -1.45, z: 0.68 } },
                                aspectmode: 'manual',
                                aspectratio: { x: 1.25, y: 1, z: 0.7 },
                            },
                        }}
                        config={PLOT_CONFIG}
                        useResizeHandler
                        style={{ width: '100%', height: '470px' }}
                    />
                    <div className="info-card">
                        Drag to rotate. The <span style={{ color: COLORS.threshold }}>amber plane</span> sits at height c.
                        The line where it cuts the sheet, dropped to the floor, is the <strong>decision boundary</strong> shown on the right.
                    </div>
                </div>
                <div className="panel">
                    <div className="panel-header">
                        <h3>2D: the input plane from above</h3>
                        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                            <Toggle checked={showMap} onChange={setShowMap}>Probability map</Toggle>
                            <Toggle checked={showGhost} onChange={setShowGhost} color={COLORS.muted}>Show c = 0.5 line</Toggle>
                        </div>
                    </div>
                    <Plot
                        data={plot2d}
                        layout={{
                            ...baseLayout(pTheme),
                            margin: { t: 20, r: 20, l: 60, b: 55 },
                            xaxis: { title: { text: 'hours studied (x₁)' }, range: copy(X1_RANGE), gridcolor: pTheme.gridColor, zeroline: false },
                            yaxis: { title: { text: 'hours slept (x₂)' }, range: copy(X2_RANGE), gridcolor: pTheme.gridColor, zeroline: false },
                            annotations: [{
                                x: 0.99, y: 0.99, xref: 'paper', yref: 'paper', xanchor: 'right', yanchor: 'top', text: tiltText, showarrow: false,
                                font: { size: 12 }, bgcolor: pTheme.labelBg, bordercolor: pTheme.gridColor, borderpad: 4,
                            }],
                        }}
                        config={PLOT_CONFIG}
                        useResizeHandler
                        style={{ width: '100%', height: '470px' }}
                    />
                    <div className="info-card">
                        <div className="formula">boundary: {w1.toFixed(2)}·x₁ + {w2.toFixed(2)}·x₂ + ({b.toFixed(2)}) = logit({c.toFixed(2)}) = {t.toFixed(2)}</div>
                        Moving the <span style={{ color: COLORS.threshold }}>threshold</span> only <strong>slides</strong> the boundary:
                        c changes the right-hand side, never the slope −w₁/w₂. Only w₁ and w₂ can <strong>turn</strong> it.
                    </div>
                </div>
            </div>

            <div className="controls glass-panel">
                <div className="sliders">
                    <Slider className="threshold-slider" label="Threshold c" value={c} min={0.01} max={0.99} step={0.01} onChange={setC} />
                    <Slider label="Weight w₁ (hours studied)" value={w1} min={-1} max={4} step={0.01} onChange={stopAnd(setW1)} />
                    <Slider label="Weight w₂ (hours slept)" value={w2} min={-1} max={4} step={0.01} onChange={stopAnd(setW2)} />
                    <Slider label="Bias b" value={b} min={-40} max={5} step={0.05} onChange={stopAnd(setB)} />
                    <div className="btn-row">
                        <button className="btn" onClick={() => trainer.setPlaying(!trainer.playing)}>
                            {trainer.playing ? <Pause size={18} /> : <Play size={18} />}
                            {trainer.playing ? 'Pause' : 'Train (gradient descent)'}
                        </button>
                        <button className="btn btn-secondary" onClick={() => { trainer.setPlaying(false); trainer.resetSteps(); setW1(0); setW2(0); setB(0); }}>
                            <RotateCcw size={16} /> Start from zero
                        </button>
                        <button className="btn btn-secondary" onClick={() => { trainer.setPlaying(false); setW1(BEST_2D.w1); setW2(BEST_2D.w2); setB(BEST_2D.b); }}>
                            <Target size={16} /> Best fit
                        </button>
                    </div>
                </div>
                <div className="metrics-grid">
                    <Metric value={`${m.nWrong} / ${STUDENTS.length}`} label="Mistakes" />
                    <Metric value={`${(100 * (1 - m.nWrong / STUDENTS.length)).toFixed(1)}%`} label="Accuracy" />
                    <Metric value={ratio === null ? '—' : ratio.toFixed(2)} label="Tilt w₁/w₂ (h of sleep per h of study)" />
                    <Metric value={`${angle.toFixed(1)}°`} label="Boundary angle to the x₁ axis" />
                    <Metric value={m.J.toFixed(3)} label="Cost J (log loss)" />
                    <Metric value={trainer.steps.toLocaleString()} label="Gradient steps (α = 0.05)" />
                </div>
            </div>
        </>
    );
}
