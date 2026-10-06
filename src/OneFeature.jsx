import React, { useMemo, useState } from 'react';
import { Play, Pause, RotateCcw, Target } from 'lucide-react';
import { STUDENTS, BEST_1D, X1_RANGE } from './data';
import { sigmoid, logit, cost, costOf, range } from './logistic';
import { Plot, COLORS, Slider, AlphaSlider, Metric, Toggle, ViewTabs, useTrainer, baseLayout, PLOT_CONFIG, copy } from './ui';
import CostPanel, { CostStatus } from './CostPanel';

const XS = STUDENTS.map((s) => [s.x1]);
const YS = STUDENTS.map((s) => s.y);
const BEST_J = costOf([BEST_1D.w], BEST_1D.b, XS, YS);

// Stack students who share the same number of hours so they stay visible on the line.
const LINE_Y = (() => {
    const seen = {};
    return STUDENTS.map((s) => {
        const k = seen[s.x1] || 0;
        seen[s.x1] = k + 1;
        return k * 0.17;
    });
})();

export default function OneFeature({ pTheme }) {
    const [w, setW] = useState(BEST_1D.w);
    const [b, setB] = useState(BEST_1D.b);
    const [c, setC] = useState(0.5);
    const [showErrors, setShowErrors] = useState(true);
    const [alpha, setAlpha] = useState(0.1);
    const [slow, setSlow] = useState(true);
    const [view, setView] = useState('line'); // right panel: 'line' (the input line) or 'cost' (the learning curve)

    const trainer = useTrainer({
        getParams: () => ({ w: [w], b }),
        setParams: (wNew, bNew) => { setW(wNew[0]); setB(bNew); },
        xs: XS, ys: YS, alpha, slow, stepsPerFrame: 80, maxSteps: 100000,
    });
    const stopAnd = (fn) => (v) => { trainer.setPlaying(false); trainer.resetRun(); fn(v); };

    const m = useMemo(() => {
        const probs = STUDENTS.map((s) => sigmoid(w * s.x1 + b));
        const preds = probs.map((p) => (p >= c ? 1 : 0));
        const wrong = STUDENTS.map((s, i) => preds[i] !== s.y);
        const nWrong = wrong.filter(Boolean).length;
        const t = logit(c);
        const xb = Math.abs(w) > 1e-9 ? (t - b) / w : null;
        return { probs, wrong, nWrong, t, xb, J: cost(probs, YS) };
    }, [w, b, c]);

    const [xLo, xHi] = X1_RANGE;
    const boundaryVisible = m.xb !== null && m.xb > xLo && m.xb < xHi;
    // Which class is predicted left of the boundary (or everywhere, if there is no boundary in view).
    const leftClass = boundaryVisible ? (w > 0 ? 0 : 1) : (sigmoid(w * 5 + b) >= c ? 1 : 0);
    const regionShapes = (y0, y1) => {
        const fill = (cls) => (cls === 1 ? COLORS.passRegion : COLORS.failRegion);
        if (!boundaryVisible) {
            return [{ type: 'rect', x0: xLo, x1: xHi, y0, y1, fillcolor: fill(leftClass), line: { width: 0 }, layer: 'below' }];
        }
        return [
            { type: 'rect', x0: xLo, x1: m.xb, y0, y1, fillcolor: fill(leftClass), line: { width: 0 }, layer: 'below' },
            { type: 'rect', x0: m.xb, x1: xHi, y0, y1, fillcolor: fill(1 - leftClass), line: { width: 0 }, layer: 'below' },
        ];
    };
    const regionLabels = (y) => {
        const name = (cls) => (cls === 1 ? 'predict pass' : 'predict fail');
        const col = (cls) => (cls === 1 ? COLORS.pass : COLORS.fail);
        if (!boundaryVisible) return [{ x: 5, y, text: name(leftClass), showarrow: false, font: { color: col(leftClass), size: 13 } }];
        return [
            { x: (xLo + m.xb) / 2, y, text: name(leftClass), showarrow: false, font: { color: col(leftClass), size: 13 } },
            { x: (m.xb + xHi) / 2, y, text: name(1 - leftClass), showarrow: false, font: { color: col(1 - leftClass), size: 13 } },
        ];
    };

    const fails = STUDENTS.map((s, i) => ({ ...s, i })).filter((s) => s.y === 0);
    const passes = STUDENTS.map((s, i) => ({ ...s, i })).filter((s) => s.y === 1);
    const errs = STUDENTS.map((s, i) => ({ ...s, i })).filter((s) => m.wrong[s.i]);
    const markerFail = { color: COLORS.fail, size: 10, symbol: 'circle', line: { color: 'white', width: 1 } };
    const markerPass = { color: COLORS.pass, size: 11, symbol: 'x', line: { color: 'white', width: 1 } };
    const markerErr = { color: COLORS.error, size: 20, symbol: 'circle-open', line: { width: 2 } };
    const xb = m.xb;
    const fmtH = (v) => (v === null ? '—' : `${v.toFixed(2)} h`);

    // ── Plot A: the sigmoid curve and the threshold line ──
    const curveX = range(xLo, xHi, 201);
    const plotA = [
        { x: curveX, y: curveX.map((x) => sigmoid(w * x + b)), mode: 'lines', type: 'scatter', line: { color: COLORS.curve, width: 3 }, name: 'f(x)', hovertemplate: 'x = %{x:.2f} h<br>f(x) = %{y:.3f}<extra></extra>' },
        { x: [xLo, xHi], y: [c, c], mode: 'lines', type: 'scatter', line: { color: COLORS.threshold, width: 2.5, dash: 'dash' }, hoverinfo: 'skip' },
        { x: fails.map((s) => s.x1), y: fails.map(() => 0), mode: 'markers', type: 'scatter', marker: markerFail, hovertemplate: 'fail · %{x} h<extra></extra>' },
        { x: passes.map((s) => s.x1), y: passes.map(() => 1), mode: 'markers', type: 'scatter', marker: markerPass, hovertemplate: 'pass · %{x} h<extra></extra>' },
    ];
    if (showErrors) plotA.push({ x: errs.map((s) => s.x1), y: errs.map((s) => s.y), mode: 'markers', type: 'scatter', marker: markerErr, hoverinfo: 'skip' });
    if (boundaryVisible) {
        plotA.push({ x: [xb, xb], y: [-0.1, 1.1], mode: 'lines', type: 'scatter', line: { color: pTheme.fontColor, width: 1.5, dash: 'dot' }, hoverinfo: 'skip' });
        plotA.push({ x: [xb], y: [c], mode: 'markers', type: 'scatter', marker: { color: COLORS.threshold, size: 13, symbol: 'diamond', line: { color: 'white', width: 1.5 } }, hovertemplate: 'f(x*) = c<extra></extra>' });
    }

    // ── Plot B: the input line ──
    const plotB = [
        { x: [xLo, xHi], y: [0, 0], mode: 'lines', type: 'scatter', line: { color: pTheme.fontColor, width: 2 }, hoverinfo: 'skip' },
        { x: fails.map((s) => s.x1), y: fails.map((s) => LINE_Y[s.i]), mode: 'markers', type: 'scatter', marker: markerFail, hovertemplate: 'fail · %{x} h<extra></extra>' },
        { x: passes.map((s) => s.x1), y: passes.map((s) => LINE_Y[s.i]), mode: 'markers', type: 'scatter', marker: markerPass, hovertemplate: 'pass · %{x} h<extra></extra>' },
    ];
    if (showErrors) plotB.push({ x: errs.map((s) => s.x1), y: errs.map((s) => LINE_Y[s.i]), mode: 'markers', type: 'scatter', marker: markerErr, hoverinfo: 'skip' });
    if (boundaryVisible) plotB.push({ x: [xb, xb], y: [-0.35, 0.75], mode: 'lines', type: 'scatter', line: { color: COLORS.threshold, width: 4 }, hoverinfo: 'skip' });

    return (
        <>
            <div className="plots-grid glass-panel">
                <div className="panel">
                    <div className="panel-header">
                        <h3>Model: f(x) = g(w·x + b)</h3>
                        <Toggle checked={showErrors} onChange={setShowErrors} color={COLORS.error}>Show mistakes</Toggle>
                    </div>
                    <Plot
                        data={plotA}
                        layout={{
                            ...baseLayout(pTheme),
                            margin: { t: 20, r: 20, l: 60, b: 55 },
                            xaxis: { title: { text: 'hours studied (x)' }, range: copy(X1_RANGE), gridcolor: pTheme.gridColor, zeroline: false },
                            yaxis: { title: { text: 'probability of passing f(x)' }, range: [-0.1, 1.12], gridcolor: pTheme.gridColor, zeroline: false, tickvals: [0, 0.25, 0.5, 0.75, 1] },
                            shapes: regionShapes(-0.1, 1.12),
                            annotations: [
                                { x: xHi, y: c, xanchor: 'right', yanchor: 'bottom', text: `threshold c = ${c.toFixed(2)}`, showarrow: false, font: { color: COLORS.threshold, size: 13 } },
                                ...(boundaryVisible ? [{ x: xb, y: 1.1, yanchor: 'top', xanchor: 'left', text: ` x* = ${xb.toFixed(2)} h`, showarrow: false, font: { size: 13 } }] : []),
                            ],
                        }}
                        config={PLOT_CONFIG}
                        useResizeHandler
                        style={{ width: '100%', height: '390px' }}
                    />
                </div>
                <div className="panel">
                    <div className="panel-header">
                        <ViewTabs
                            options={[{ key: 'line', label: 'The input line' }, { key: 'cost', label: 'Cost J while training' }]}
                            value={view} onChange={setView}
                        />
                        {view === 'cost' && <CostStatus trainer={trainer} J={m.J} />}
                    </div>
                    {view === 'cost' ? (
                        <CostPanel trainer={trainer} J={m.J} bestJ={BEST_J} minSpan={slow ? 30 : 2000} height="300px" pTheme={pTheme} />
                    ) : (
                        <>
                            <Plot
                                data={plotB}
                                layout={{
                                    ...baseLayout(pTheme),
                                    margin: { t: 6, r: 20, l: 20, b: 55 },
                                    xaxis: { title: { text: 'hours studied (x)' }, range: copy(X1_RANGE), gridcolor: pTheme.gridColor, zeroline: false },
                                    yaxis: { range: [-0.4, 0.95], visible: false, fixedrange: true },
                                    shapes: regionShapes(-0.35, 0.75),
                                    annotations: [
                                        ...regionLabels(-0.25),
                                        ...(boundaryVisible ? [{ x: xb, y: 0.75, yanchor: 'bottom', text: `boundary x* = ${xb.toFixed(2)} h`, showarrow: false, font: { color: COLORS.threshold, size: 13 } }] : []),
                                    ],
                                }}
                                config={{ ...PLOT_CONFIG, displayModeBar: false }}
                                useResizeHandler
                                style={{ width: '100%', height: '190px' }}
                            />
                            <div className="info-card">
                                With one feature, the inputs live on a line and the decision boundary is a <strong>single point</strong> x*:
                                the place where the curve reaches the threshold, f(x*) = c.
                                <div className="formula">w·x* + b = logit(c) = ln(c / (1 − c))&nbsp;&nbsp;→&nbsp;&nbsp;x* = (logit(c) − b) / w</div>
                                <div className="formula">
                                    {Math.abs(w) > 1e-9
                                        ? `x* = (${m.t.toFixed(2)} − (${b.toFixed(2)})) / ${w.toFixed(2)} = ${fmtH(xb)}`
                                        : 'w = 0: the curve is flat, so there is no boundary'}
                                </div>
                                Move the <span style={{ color: COLORS.threshold }}>threshold</span>: the point slides along the line.
                                With one feature, a cut-off can put the boundary anywhere.
                            </div>
                        </>
                    )}
                </div>
            </div>

            <div className="controls glass-panel">
                <div className="sliders">
                    <Slider className="threshold-slider" label="Threshold c" value={c} min={0.01} max={0.99} step={0.01} onChange={setC} />
                    <Slider label="Weight w" value={w} min={-1} max={4} step={0.01} onChange={stopAnd(setW)} />
                    <Slider label="Bias b" value={b} min={-20} max={5} step={0.05} onChange={stopAnd(setB)} />
                    <AlphaSlider alpha={alpha} onChange={setAlpha} />
                    <div className="btn-row">
                        <button className="btn" onClick={() => trainer.setPlaying(!trainer.playing)}>
                            {trainer.playing ? <Pause size={18} /> : <Play size={18} />}
                            {trainer.playing ? 'Pause' : 'Train (gradient descent)'}
                        </button>
                        <button className="btn btn-secondary" onClick={() => { trainer.setPlaying(false); trainer.resetRun(); setW(0); setB(0); }}>
                            <RotateCcw size={16} /> Start from w = b = 0
                        </button>
                        <button className="btn btn-secondary" onClick={() => { trainer.setPlaying(false); trainer.resetRun(); setW(BEST_1D.w); setB(BEST_1D.b); }}>
                            <Target size={16} /> Best fit
                        </button>
                        <Toggle checked={slow} onChange={setSlow}>Slow motion</Toggle>
                    </div>
                </div>
                <div className="metrics-grid">
                    <Metric value={fmtH(xb)} label="Boundary x*" />
                    <Metric value={`${m.nWrong} / ${STUDENTS.length}`} label="Mistakes" />
                    <Metric value={`${(100 * (1 - m.nWrong / STUDENTS.length)).toFixed(1)}%`} label="Accuracy" />
                    <Metric value={m.J.toFixed(3)} label="Cost J (log loss)" />
                    <Metric value={trainer.steps.toLocaleString()} label={`Gradient steps (α = ${alpha})`} />
                    <Metric value={m.t.toFixed(2)} label="logit(c)" />
                </div>
            </div>
        </>
    );
}
