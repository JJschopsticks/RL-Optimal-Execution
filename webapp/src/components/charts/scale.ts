// scale.ts
//
// Port of frontend/index.html's chart math (scale/niceTicks/padSeries/
// pathFor/pathForHeld) as pure, framework-agnostic functions -- unchanged
// from the original vanilla-JS logic, just typed. Kept pure (no DOM/React)
// so LiveSessionView (a growing series, no padding) and SessionDetailView (a
// finished series, dashed-tail padding) can share the exact same functions.

import type { TickRecord } from "../../types";

export const MARGIN = { top: 10, right: 14, bottom: 20, left: 46 };
export const VB_W = 900;
export const VB_H = 200;

export type ScaleFn = (v: number) => number;

export function scale(domain: [number, number], range: [number, number]): ScaleFn {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

export interface PaddedPoint {
  v: number;
  real: boolean;
}

/** Holds the last value flat past a finished episode's natural length --
 * used by the (static) SessionDetailView. LiveSessionView instead just grows
 * its own array and never calls this. */
export function padSeries(trace: TickRecord[], xMax: number, key: keyof TickRecord): PaddedPoint[] {
  const out: PaddedPoint[] = new Array(xMax + 1);
  let last = 0;
  for (let i = 0; i <= xMax; i++) {
    if (i < trace.length) {
      last = trace[i][key] as number;
      out[i] = { v: last, real: true };
    } else {
      out[i] = { v: last, real: false };
    }
  }
  return out;
}

export function pathFor(xScale: ScaleFn, yScale: ScaleFn, arr: PaddedPoint[], realOnly: boolean): string {
  let d = "";
  let started = false;
  for (let i = 0; i < arr.length; i++) {
    if (realOnly && !arr[i].real) continue;
    const x = xScale(i);
    const y = yScale(arr[i].v);
    d += (started ? "L" : "M") + x.toFixed(2) + " " + y.toFixed(2) + " ";
    started = true;
  }
  return d.trim();
}

/** Dashed continuation from the last real point to the end of the domain. */
export function pathForHeld(xScale: ScaleFn, yScale: ScaleFn, arr: PaddedPoint[]): string {
  let lastReal = -1;
  for (let i = 0; i < arr.length; i++) if (arr[i].real) lastReal = i;
  if (lastReal < 0 || lastReal >= arr.length - 1) return "";
  let d = "M" + xScale(lastReal).toFixed(2) + " " + yScale(arr[lastReal].v).toFixed(2) + " ";
  d += "L" + xScale(arr.length - 1).toFixed(2) + " " + yScale(arr[arr.length - 1].v).toFixed(2);
  return d;
}

export function niceTicks(domain: [number, number], count: number): number[] {
  const span = domain[1] - domain[0];
  const step = span / count;
  const out: number[] = [];
  for (let i = 0; i <= count; i++) out.push(domain[0] + i * step);
  return out;
}
