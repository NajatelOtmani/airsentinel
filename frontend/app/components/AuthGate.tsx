"use client";
import { useEffect, useState } from "react";
import { api, getToken } from "@/lib/api";

export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [ready, setReady] = useState(false);
  const [authed, setAuthed] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    setAuthed(!!getToken());
    setReady(true);
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      await api.login(username, password);
      setAuthed(true);
    } catch {
      setError("Invalid credentials");
    }
  };

  if (!ready) return null;

  if (!authed) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0F1D22]">
        <form onSubmit={handleLogin} className="bg-white/5 p-8 rounded-2xl w-80 flex flex-col gap-4">
          <h1 className="text-white text-lg font-semibold">AirSentinel Login</h1>
          <input
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="bg-[#1A1D21] text-white rounded-lg px-3 py-2 text-sm outline-none border border-white/10 focus:border-teal-500"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="bg-[#1A1D21] text-white rounded-lg px-3 py-2 text-sm outline-none border border-white/10 focus:border-teal-500"
          />
          {error && <p className="text-red-400 text-xs">{error}</p>}
          <button className="bg-teal-600 hover:bg-teal-500 text-white rounded-lg py-2 text-sm font-medium transition-colors">
            Log in
          </button>
        </form>
      </div>
    );
  }

  return <>{children}</>;
}