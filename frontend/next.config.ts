import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required for AWS Amplify SSR deployment
  output: "standalone",
};

export default nextConfig;
