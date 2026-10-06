import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
import { gdStep, costOf } from './logistic';

// Plotly.js 3+ can crash in React 19 while unmounting; guard its purge method.
const originalPurge = Plotly.purge;
Plotly.purge = (gd) => {
    try {
        if (originalPurge) originalPurge(gd);
    } catch (e) {
        console.warn('Caught Plotly purge error on unmount:', e);
    }
};

export const Plot = createPlotlyComponent(Plotly);

// Plotly mutates the arrays inside `layout` (and may touch `data`) when the user zooms or pans.
// Every shared array passed to a plot goes through this, so the originals never change.
export const copy = (arr) => [...arr];

export const COLORS = {
    fail: '#3b82f6',
    pass: '#f97316',
    threshold: '#f59e0b',
    curve: '#10b981',
    cost: '#8b5cf6',
    error: '#ef4444',
    muted: '#94a3b8',
    failRegion: 'rgba(59, 130, 246, 0.12)',
    passRegion: 'rgba(249, 115, 22, 0.12)',
};

export function Slider({ label, value, min, max, step, onChange, format = (v) => v.toFixed(2), className = '' }) {
    return (
        <div className={`slider-container ${className}`}>
            <div className="slider-header">
                <span>{label}</span>
                <span style={{ color: className.includes('threshold') ? COLORS.threshold : 'var(--primary)', fontFamily: 'monospace' }}>
                    {format(value)}
                </span>
            </div>
            <input type="range" min={min} max={max} step={step} value={value}
                onChange={(e) => onChange(Number(e.target.value))} />
        </div>
    );
}

export function Metric({ value, label }) {
    return (
        <div className="metric-card">
            <div className="metric-value">{value}</div>
            <div className="metric-label" style={{ textTransform: 'none' }}>{label}</div>
        </div>
    );
}

export function Toggle({ checked, onChange, children, color = '#3b82f6' }) {
    return (
        <label className="toggle">
            <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} style={{ accentColor: color }} />
            {children}
        </label>
    );
}

// Runs gradient descent in an animation loop. params = { w: [...], b }.
// `history` is the learning curve of the current run: [step, J] for the starting model and after every frame.
export function useTrainer({ getParams, setParams, xs, ys, alpha, stepsPerFrame, maxSteps }) {
    const [playing, setPlaying] = useState(false);
    const [steps, setSteps] = useState(0);
    const [history, setHistory] = useState([]);
    const [status, setStatus] = useState(null); // 'converged' or 'limit' once a run stops on its own
    const stepsRef = useRef(0);
    const historyRef = useRef([]);
    const paramsRef = useRef(null);

    useEffect(() => {
        if (!playing) return undefined;
        paramsRef.current = getParams();
        if (!historyRef.current.length) {
            const { w, b } = paramsRef.current;
            historyRef.current = [[stepsRef.current, costOf(w, b, xs, ys)]];
            setHistory(historyRef.current);
        }
        setStatus(null);
        let raf;
        const loop = () => {
            let { w, b } = paramsRef.current;
            let grad = Infinity;
            for (let k = 0; k < stepsPerFrame; k++) {
                const r = gdStep(w, b, xs, ys, alpha);
                w = r.w;
                b = r.b;
                grad = r.grad;
            }
            paramsRef.current = { w, b };
            stepsRef.current += stepsPerFrame;
            historyRef.current = [...historyRef.current, [stepsRef.current, costOf(w, b, xs, ys)]];
            setParams(w, b);
            setSteps(stepsRef.current);
            setHistory(historyRef.current);
            if (grad < 5e-5 || stepsRef.current >= maxSteps) {
                setStatus(grad < 5e-5 ? 'converged' : 'limit');
                setPlaying(false);
                return;
            }
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [playing]);

    // Forget the current run (step count and learning curve). Called whenever w or b change by hand,
    // so the next run starts a new curve from the model on the sliders.
    const resetRun = () => {
        stepsRef.current = 0;
        historyRef.current = [];
        setSteps(0);
        setHistory([]);
        setStatus(null);
    };
    return { playing, setPlaying, steps, history, status, resetRun };
}

// Base Plotly layout shared by the 2D plots.
export const baseLayout = (pTheme) => ({
    autosize: true,
    showlegend: false,
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: pTheme.fontColor, family: 'Inter, system-ui, sans-serif' },
    uirevision: 'keep',
});

export const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };
