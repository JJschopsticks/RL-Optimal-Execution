// App.tsx
import { NavLink, Route, Routes } from "react-router-dom";
import { LiveSessionView } from "./views/LiveSessionView";
import { SessionHistoryView } from "./views/SessionHistoryView";
import { SessionDetailView } from "./views/SessionDetailView";
import { ConnectionStatusDot } from "./components/ConnectionStatusDot";

export default function App() {
  return (
    <div className="page">
      <header className="masthead">
        <ConnectionStatusDot />
        <h1>Smart Order Router — Live Paper Trading</h1>
        <p className="sub">
          All five policies run concurrently against the live Binance order book, fully simulated -- no real orders,
          no exchange API keys.
        </p>
        <nav className="nav">
          <NavLink to="/live" className={({ isActive }) => (isActive ? "active" : "")}>
            Live
          </NavLink>
          <NavLink to="/history" className={({ isActive }) => (isActive ? "active" : "")}>
            History
          </NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<LiveSessionView />} />
        <Route path="/live" element={<LiveSessionView />} />
        <Route path="/history" element={<SessionHistoryView />} />
        <Route path="/history/:sessionId" element={<SessionDetailView />} />
      </Routes>
    </div>
  );
}
