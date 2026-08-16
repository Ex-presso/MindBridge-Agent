import type { Metadata } from "next";
import "./globals.css";
import { Toaster } from "sonner";
import { AuthInitializer } from "@/components/AuthInitializer";

export const metadata: Metadata = {
  title: "MindBridge",
  description: "A compassionate mental health companion",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full antialiased bg-background text-foreground">
        <AuthInitializer />
        {children}
        <Toaster position="top-right" richColors />
      </body>
    </html>
  );
}
