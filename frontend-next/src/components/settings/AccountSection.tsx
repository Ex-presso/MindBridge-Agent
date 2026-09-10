"use client";
import { useState } from "react";
import { Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/useAuth";
import { useMemory } from "@/hooks/useMemory";

export function AccountSection() {
  const { user, deleteAccount } = useAuth();
  const { refreshMemory } = useMemory();
  const [accountDialogOpen, setAccountDialogOpen] = useState(false);
  const [accountPassword, setAccountPassword] = useState("");
  const [accountBusy, setAccountBusy] = useState(false);

  const handleDeleteAccount = async () => {
    if (!accountPassword) return;
    setAccountBusy(true);
    try {
      await deleteAccount(accountPassword);
      toast.success("Account deleted");
    } catch (err) {
      setAccountBusy(false);
      await refreshMemory().catch(() => undefined);
      toast.error(err instanceof Error ? err.message : "Failed to delete account");
    }
  };

  return (
    <section aria-labelledby="account-settings-title">
      <div className="mb-4">
        <h2 id="account-settings-title" className="text-base font-semibold">Account</h2>
        <p className="mt-1 text-xs leading-5 text-(--muted-foreground)">
          Your profile and account controls.
        </p>
      </div>

      <div className="rounded-xl border border-(--border) bg-(--background) p-5">
        <div className="flex flex-col gap-3">
          <div>
            <Label className="text-xs">Display name</Label>
            <Input value={user?.display_name ?? ""} disabled className="mt-1 text-xs" />
          </div>
          <div>
            <Label className="text-xs">Email</Label>
            <Input value={user?.email ?? ""} disabled className="mt-1 text-xs" />
          </div>
          <div>
            <Label className="text-xs">Password</Label>
            <Input value="••••••••" type="password" disabled className="mt-1 text-xs" />
          </div>
          <p className="text-xs leading-5 text-(--muted-foreground)">
            Editing your profile, email, and password is not supported by the API yet.
          </p>
        </div>
      </div>

      <div className="mt-6 rounded-xl border border-(--destructive)/40 bg-(--background) p-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm font-medium text-(--destructive)">Delete account</p>
            <p className="mt-1 text-xs leading-5 text-(--muted-foreground)">
              Permanently removes your conversations, checkpoints, API keys, and all stored memory. This cannot be undone.
            </p>
          </div>

          <Dialog open={accountDialogOpen} onOpenChange={(open) => {
            setAccountDialogOpen(open);
            if (!open && !accountBusy) setAccountPassword("");
          }}>
            <DialogTrigger asChild>
              <Button size="sm" variant="destructive" className="shrink-0">
                <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                Delete account
              </Button>
            </DialogTrigger>
            <DialogContent aria-describedby="delete-account-description">
              <DialogHeader>
                <DialogTitle>Delete your account permanently?</DialogTitle>
                <p id="delete-account-description" className="text-sm leading-6 text-(--muted-foreground)">
                  Enter your password to confirm. If external cleanup is temporarily unavailable, your account stays disabled and the deletion can be retried.
                </p>
              </DialogHeader>
              <div>
                <Label htmlFor="delete-account-password">Password</Label>
                <Input
                  id="delete-account-password"
                  type="password"
                  autoComplete="current-password"
                  value={accountPassword}
                  onChange={(event) => setAccountPassword(event.target.value)}
                  disabled={accountBusy}
                  className="mt-1"
                />
              </div>
              <DialogFooter>
                <Button
                  variant="outline"
                  onClick={() => setAccountDialogOpen(false)}
                  disabled={accountBusy}
                >
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  onClick={handleDeleteAccount}
                  disabled={accountBusy || !accountPassword}
                >
                  {accountBusy ? "Deleting…" : "Delete permanently"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>
    </section>
  );
}
