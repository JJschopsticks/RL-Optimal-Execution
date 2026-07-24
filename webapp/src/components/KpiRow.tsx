// KpiRow.tsx
export interface KpiTile {
  label: string;
  value: string;
  sub?: string;
  hero?: boolean;
}

export function KpiRow({ tiles }: { tiles: KpiTile[] }) {
  return (
    <section className="kpi-row">
      {tiles.map((t) => (
        <div className={`kpi${t.hero ? " hero" : ""}`} key={t.label}>
          <div className="label">{t.label}</div>
          <div className="value">{t.value}</div>
          {t.sub && <div className="foot">{t.sub}</div>}
        </div>
      ))}
    </section>
  );
}
