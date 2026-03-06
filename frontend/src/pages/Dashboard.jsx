import { useEffect, useState, useCallback } from "react";
import StatCards       from "../components/StatCards";
import EmployeeTable   from "../components/EmployeeTable";
import TrendChart      from "../components/TrendChart";
import DepartmentChart from "../components/DepartmentChart";
import CheckHistory    from "../components/CheckHistory";

const API = "http://localhost:5000";

export default function Dashboard({ latestUpdate }) {
  const [stats,       setStats]       = useState(null);
  const [employees,   setEmployees]   = useState([]);
  const [trend,       setTrend]       = useState([]);
  const [departments, setDepartments] = useState([]);
  const [checks,      setChecks]      = useState([]);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [loading,     setLoading]     = useState(true);

  const fetchAll = useCallback(async () => {
    try {
      const [statsRes, empRes, trendRes, deptRes, checksRes] = await Promise.all([
        fetch(`${API}/api/stats`),
        fetch(`${API}/api/employees/status`),
        fetch(`${API}/api/trend`),
        fetch(`${API}/api/departments`),
        fetch(`${API}/api/checks?limit=30`)
      ]);

      const [statsData, empData, trendData, deptData, checksData] = await Promise.all([
        statsRes.json(),
        empRes.json(),
        trendRes.json(),
        deptRes.json(),
        checksRes.json()
      ]);

      setStats(statsData);
      setEmployees(empData);
      setTrend(trendData);
      setDepartments(deptData);
      setChecks(checksData);
      setLastRefresh(new Date().toLocaleTimeString());
      setLoading(false);
    } catch (err) {
      console.error("Fetch error:", err);
      setLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  // Refresh when WebSocket update arrives
  useEffect(() => {
    if (latestUpdate) {
      fetchAll();
    }
  }, [latestUpdate, fetchAll]);

  // Auto refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(fetchAll, 30000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  if (loading) {
    return (
      <div
        className="flex items-center justify-center h-screen"
        style={{ fontFamily: "var(--font-mono)", color: "var(--amber)" }}
      >
        <div className="text-center">
          <div className="text-2xl mb-3 animate-blink">⬡</div>
          <p>Connecting to IndustriGuard backend...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-screen-2xl mx-auto">
      {/* Last refresh indicator */}
      {lastRefresh && (
        <p
          className="text-xs mb-4"
          style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}
        >
          Last updated: {lastRefresh}
          {latestUpdate && (
            <span style={{ color: "var(--green)" }}>
              {" "}· Live update received from {latestUpdate.employee_id}
            </span>
          )}
        </p>
      )}

      {/* Stat cards */}
      <StatCards stats={stats} />

      {/* Employee status table — full width */}
      <EmployeeTable employees={employees} latestUpdate={latestUpdate} />

      {/* Trend chart */}
      <TrendChart trend={trend} />

      {/* Department chart */}
      <DepartmentChart departments={departments} />

      {/* Check history */}
      <CheckHistory checks={checks} />
    </div>
  );
}