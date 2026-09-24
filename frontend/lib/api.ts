const API_BASE = "http://localhost:8000";

export function getToken() {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("airsentinel_token");
}

export function setToken(token: string) {
  localStorage.setItem("airsentinel_token", token);
}

export function clearToken() {
  localStorage.removeItem("airsentinel_token");
}

async function request(path: string, options: RequestInit = {}) {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  login: async (username: string, password: string) => {
    const body = new URLSearchParams();
    body.append("username", username);
    body.append("password", password);

    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: body.toString(),
    });
    if (!res.ok) throw new Error("Invalid credentials");
    const data = await res.json();
    setToken(data.access_token);
    return data;
  },
  getSensors: () => request("/api/v1/sensors/"),
  getSensorLatest: (id: string) => request(`/api/v1/sensors/${id}/latest`),
  getAnomalies: (id: string, limit = 100) =>
    request(`/api/v1/anomalies/${id}?limit=${limit}`),
  getForecast: (id: string) => request(`/api/v1/forecasts/${id}`),
  askAgent: (query: string) =>
    request("/api/v1/agent/", { method: "POST", body: JSON.stringify({ query }) }),
  generateReport: (id: string) =>
    request(`/api/v1/reports/${id}`, { method: "POST" }),
  getHealth: () => request("/health"),
};