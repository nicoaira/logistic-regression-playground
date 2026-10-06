import React from 'react';
import { Plot, COLORS, baseLayout, PLOT_CONFIG } from './ui';

const STATUS_TEXT = { converged: 'converged: the slopes are almost 0', limit: 'step limit reached' };

// The learning curve: cost J against gradient steps, drawn live while the model trains.
// history: [step, J] pairs of the current run; J: cost of the model on the sliders now;
// bestJ: cost of the best fit (the bottom of the bowl); minSpan: steps shown before the axis starts to grow.
export default function CostPanel({ history, steps, J, bestJ, minSpan, alpha, playing, status, pTheme }) {
    const xMax = Math.max(minSpan, steps * 1.05);
    const yMax = 1.08 * Math.max(0.8, J, ...history.map((p) => p[1]));
    const badge = playing ? 'training…' : status ? STATUS_TEXT[status] : steps > 0 ? 'paused' : null;

    const data = [
        {
            x: history.map((p) => p[0]), y: history.map((p) => p[1]), mode: 'lines', type: 'scatter',
            line: { color: COLORS.cost, width: 3 }, hovertemplate: 'step %{x:,}<br>J = %{y:.4f}<extra></extra>',
        },
        {
            x: [steps], y: [J], mode: 'markers', type: 'scatter', cliponaxis: false,
            marker: { color: COLORS.cost, size: 12, line: { color: 'white', width: 1.5 } },
            hovertemplate: 'current model<br>step %{x:,}<br>J = %{y:.4f}<extra></extra>',
        },
    ];

    return (
        <div className="cost-panel glass-panel">
            <div className="panel">
                <div className="panel-header">
                    <h3>Cost J while training</h3>
                    <div className="cost-readout">
                        {badge && <span className="state-badge">{badge}</span>}
                        <span className="cost-value" style={{ color: COLORS.cost }}>J = {J.toFixed(4)}</span>
                    </div>
                </div>
                <Plot
                    data={data}
                    layout={{
                        ...baseLayout(pTheme),
                        margin: { t: 20, r: 20, l: 60, b: 55 },
                        xaxis: { title: { text: `gradient steps (α = ${alpha})` }, range: [0, xMax], gridcolor: pTheme.gridColor, zeroline: false },
                        yaxis: { title: { text: 'cost J (log loss)' }, range: [0, yMax], gridcolor: pTheme.gridColor, zeroline: false },
                        shapes: [{
                            type: 'line', xref: 'paper', x0: 0, x1: 1, y0: bestJ, y1: bestJ,
                            line: { color: COLORS.muted, width: 2, dash: 'dash' },
                        }],
                        annotations: [
                            {
                                xref: 'paper', x: 1, y: bestJ, xanchor: 'right', yanchor: 'top', showarrow: false,
                                text: `lowest possible J (best fit) = ${bestJ.toFixed(3)}`, font: { color: COLORS.muted, size: 12 },
                            },
                            ...(history.length ? [] : [{
                                xref: 'paper', yref: 'paper', x: 0.5, y: 0.75, showarrow: false,
                                text: 'Press Train: the curve is drawn here as gradient descent runs', font: { color: COLORS.muted, size: 13 },
                            }]),
                        ],
                    }}
                    config={PLOT_CONFIG}
                    useResizeHandler
                    style={{ width: '100%', height: '320px' }}
                />
                <div className="info-card">
                    Each point is the cost J after that many gradient steps; the <span style={{ color: COLORS.cost }}>dot</span> is
                    the model on the sliders now. J averages the log loss over the 40 students, so it uses the probabilities f, not
                    the pass/fail decisions: moving the <span style={{ color: COLORS.threshold }}>threshold</span> changes the mistakes
                    but never J. With all parameters at 0, every student gets f = 0.5 and J = ln 2 ≈ 0.693. Each step goes
                    downhill, so the curve only falls, and it flattens as it nears the bottom of the bowl (dashed line).
                </div>
            </div>
        </div>
    );
}
