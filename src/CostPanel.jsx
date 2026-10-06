import React from 'react';
import { Plot, COLORS, baseLayout, PLOT_CONFIG } from './ui';

const STATUS_TEXT = { converged: 'converged: the slopes are almost 0', limit: 'step limit reached' };

// Badge and current cost, for the header of the panel that shows the learning curve.
export function CostStatus({ trainer, J }) {
    const { playing, status, steps } = trainer;
    const badge = playing ? 'training…' : status ? STATUS_TEXT[status] : steps > 0 ? 'paused' : null;
    return (
        <div className="cost-readout">
            {badge && <span className="state-badge">{badge}</span>}
            <span className="cost-value" style={{ color: COLORS.cost }}>J = {J.toFixed(4)}</span>
        </div>
    );
}

// The learning curve: cost J against gradient steps, drawn live while the model trains.
// J: cost of the model on the sliders now; bestJ: cost of the best fit (the bottom of the bowl);
// minSpan: steps shown before the axis starts to grow.
export default function CostPanel({ trainer, J, bestJ, minSpan, height, pTheme }) {
    const { history, steps } = trainer;
    const xMax = Math.max(minSpan, steps * 1.05);
    const yMax = 1.08 * Math.max(0.8, J, ...history.map((p) => p[1]));
    // Points one step apart (slow motion) get a dot each, shrinking away once there are too many to tell apart.
    const dot = Math.max(0, 7 - history.length / 40);
    const isSingleStep = (i) => [i - 1, i + 1].some((k) => history[k] && Math.abs(history[k][0] - history[i][0]) === 1);

    const data = [
        {
            x: history.map((p) => p[0]), y: history.map((p) => p[1]), mode: 'lines+markers', type: 'scatter',
            line: { color: COLORS.cost, width: 3 }, marker: { color: COLORS.cost, size: history.map((_, i) => (isSingleStep(i) ? dot : 0)) },
            hovertemplate: 'step %{x:,}<br>J = %{y:.4f}<extra></extra>',
        },
        {
            x: [steps], y: [J], mode: 'markers', type: 'scatter', cliponaxis: false,
            marker: { color: COLORS.cost, size: 12, line: { color: 'white', width: 1.5 } },
            hovertemplate: 'current model<br>step %{x:,}<br>J = %{y:.4f}<extra></extra>',
        },
    ];

    return (
        <>
            <Plot
                data={data}
                layout={{
                    ...baseLayout(pTheme),
                    margin: { t: 20, r: 20, l: 60, b: 55 },
                    xaxis: { title: { text: 'gradient steps' }, range: [0, xMax], gridcolor: pTheme.gridColor, zeroline: false },
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
                style={{ width: '100%', height }}
            />
            <div className="info-card">
                Each point is J after that many gradient steps; the <span style={{ color: COLORS.cost }}>dot</span> is the model
                on the sliders now. J uses the probabilities f, not the pass/fail decisions, so
                the <span style={{ color: COLORS.threshold }}>threshold</span> never changes it. With a small α every step goes
                downhill and the curve settles on the lowest possible cost (dashed line). With too large an α the steps
                overshoot the bottom of the bowl and J jumps up and down.
            </div>
        </>
    );
}
