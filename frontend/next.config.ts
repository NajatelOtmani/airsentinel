// frontend/next.config.ts (or next.config.js)
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // <--- ADD THIS LINE
};

export default nextConfig;