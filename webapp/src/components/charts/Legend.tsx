// Legend.tsx
import { POLICY_COLOR_VAR } from "../../types";

export function colorVar(name: string): string {
  return `var(${POLICY_COLOR_VAR[name] ?? "--muted"})`;
}

export function Legend({ names }: { names: string[] }) {
  return (
    <div className="legend">
      {names.map((name) => (
        <div className="item" key={name}>
          <span className="swatch" style={{ background: colorVar(name) }} />
          <span>{name}</span>
        </div>
      ))}
    </div>
  );
}
