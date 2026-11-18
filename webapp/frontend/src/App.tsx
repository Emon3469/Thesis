import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { motion } from "framer-motion";
import Explorer from "./pages/Explorer";
import Compare from "./pages/Compare";

export default function App() {
  const [health, setHealth] = useState<"wait" | "ok" | "bad">("wait");
  useEffect(() => {
    const check = () => fetch("/api/health").then((r) => r.json()).then((h) => setHealth(h.ok && h.frozen_md5_verified ? "ok" : "bad")).catch(() => setHealth("bad"));
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);
  return (
    <>
      <header className="topbar">
        <NavLink to="/" className="brand"><i /><b>BanglaMeta</b><span>Bangla Word2Vec meta-embedding</span></NavLink>
        <nav className="nav">
          {[["/", "Explorer"], ["/compare", "Compare models"]].map(([to, label]) => (
            <NavLink key={to} to={to} end className={({ isActive }) => (isActive ? "active" : "")}>
              {({ isActive }) => (<>{isActive && <motion.span layoutId="pill" className="pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}<span>{label}</span></>)}
            </NavLink>
          ))}
        </nav>
        <div className="status" title="model server status; the frozen merged model's checksum is verified at start-up">
          <span className={`dot ${health === "ok" ? "ok" : health === "bad" ? "bad" : ""}`} />
          <span>{health === "ok" ? "merged model loaded · checksum verified" : health === "bad" ? "model server offline" : "connecting…"}</span>
        </div>
      </header>
      <Routes>
        <Route path="/" element={<Explorer />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}
