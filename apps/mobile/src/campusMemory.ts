import { getApiUrl } from './vision';
import { getSightDeviceId, getSightHeaders, getSightSessionId } from './sightIdentity';
import type { LocalDetection } from './detector.types';

export type CampusAlert = {
  id: number;
  location: string;
  category: 'hazard' | 'obstacle' | 'sign' | 'room_status';
  label: string;
  description: string;
  severity: 'low' | 'medium' | 'high';
  reported_by: string;
  created_at: string;
  relative_time: string;
  is_active: boolean;
};

export type WhatChangedResult = {
  has_changed: boolean;
  delta_summary: string;
  speech: string;
  added_objects: string[];
  removed_objects: string[];
  moved_objects: string[];
};

export type BridgeMessage = {
  id: number;
  session_id: string;
  sender_role: 'deaf' | 'blind';
  text: string;
  created_at: string;
  relative_time: string;
};

export type SightObservation = {
  id: number;
  category: string;
  label: string;
  description: string;
  zone_id: string;
  timestamp: string;
  source_device: string;
  source_session: string;
  confidence: number;
  verification_status: 'verified' | 'unverified';
  privacy_scope: 'personal' | 'shared';
  room_x: number | null;
  room_y: number | null;
  expires_at: string;
  origin: 'manual' | 'detector';
};

export type SightDashboard = {
  zone_id: string;
  connected_devices: number;
  recent_observations: SightObservation[];
  tracking_latency_ms: number | null;
  tracking_requests_last_5m: number;
};

export type SightChanges = {
  zone_id: string;
  generated_at: string;
  compared_seconds: number;
  added: string[];
  previously_seen: string[];
  summary: string;
  verified_only: boolean;
};

export const currentSightSession = getSightSessionId;
export const currentSightDevice = getSightDeviceId;

async function sightJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const apiUrl = getApiUrl();
  if (!apiUrl) throw new Error('Connect the DGX Spark API to use Shared Sight');
  const headers = new Headers(getSightHeaders());
  new Headers(init.headers).forEach((value, key) => headers.set(key, value));
  const response = await fetch(`${apiUrl}${path}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail || `Shared Sight request failed (${response.status})`);
  }
  return await response.json() as T;
}

export async function submitSightObservation(input: {
  zone_id: string;
  category: string;
  label: string;
  description?: string;
  confidence: number;
  privacy_scope: 'personal' | 'shared';
  verified: boolean;
  room_x?: number;
  room_y?: number;
  origin?: 'manual' | 'detector';
}): Promise<SightObservation> {
  return sightJson<SightObservation>('/v1/observations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
}

export async function getSightDashboard(zoneId: string): Promise<SightDashboard> {
  return sightJson<SightDashboard>(`/v1/shared-sight/dashboard?zone_id=${encodeURIComponent(zoneId)}`);
}

export async function getSightChanges(zoneId: string): Promise<SightChanges> {
  return sightJson<SightChanges>(`/v1/observations/changes?zone_id=${encodeURIComponent(zoneId)}`);
}

export async function searchPersonalRecall(query: string): Promise<SightObservation[]> {
  return sightJson<SightObservation[]>(`/v1/personal/observations?query=${encodeURIComponent(query)}`);
}

export async function removeSightObservation(id: number): Promise<void> {
  await sightJson<{ status: string }>(`/v1/observations/${id}`, { method: 'DELETE' });
}

export const CAMPUS_LOCATIONS = [
  'Library Corridor',
  'Hallway 200',
  'Room 204',
  'Elevator B',
  'Main Campus Lobby',
  'Cafeteria',
] as const;

export type CampusLocation = (typeof CAMPUS_LOCATIONS)[number];

const SEED_ALERTS: CampusAlert[] = [];

const LOCAL_STORAGE_KEY_ALERTS = 'guidelens_campus_alerts_v2';
const LOCAL_STORAGE_KEY_BASELINES = 'guidelens_room_baselines';
const LOCAL_STORAGE_KEY_BRIDGE = 'guidelens_bridge_messages';

const inMemoryCache: Record<string, string> = {};

function getStoredJson<T>(key: string, fallback: T): T {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const raw = window.localStorage.getItem(key);
      if (raw) return JSON.parse(raw) as T;
    }
  } catch {}
  if (inMemoryCache[key]) {
    try {
      return JSON.parse(inMemoryCache[key]) as T;
    } catch {}
  }
  return fallback;
}

function setStoredJson<T>(key: string, value: T): void {
  const serialized = JSON.stringify(value);
  inMemoryCache[key] = serialized;
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(key, serialized);
    }
  } catch {}
}

/**
 * Fetch active campus alerts.
 * Gracefully merges server state with local storage.
 * If server is unreachable or offline, returns local alerts with ZERO errors.
 */
export async function getCampusAlerts(location?: string): Promise<CampusAlert[]> {
  const localAlerts = getStoredJson<CampusAlert[]>(LOCAL_STORAGE_KEY_ALERTS, SEED_ALERTS);
  const apiUrl = getApiUrl();

  if (!apiUrl) {
    return filterAlerts(localAlerts, location);
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const query = location ? `?location=${encodeURIComponent(location)}` : '';
    const res = await fetch(`${apiUrl}/v1/memory/alerts${query}`, {
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...getSightHeaders() },
    });
    clearTimeout(timeout);

    if (res.ok) {
      const serverAlerts = (await res.json()) as CampusAlert[];
      if (Array.isArray(serverAlerts) && serverAlerts.length > 0) {
        setStoredJson(LOCAL_STORAGE_KEY_ALERTS, serverAlerts);
        return filterAlerts(serverAlerts, location);
      }
    }
  } catch {
    // Offline or connection refused: silently fallback to local storage
  }

  return filterAlerts(localAlerts, location);
}

function filterAlerts(alerts: CampusAlert[], location?: string): CampusAlert[] {
  const active = alerts.filter((a) => a.is_active);
  if (!location || location.trim() === '' || location === 'All Campus') {
    return active;
  }
  const needle = location.toLowerCase();
  return active.filter(
    (a) => a.location.toLowerCase().includes(needle) || a.location === 'All Campus'
  );
}

/**
 * Report a new obstacle, hazard, or room status from this device.
 * Immediately saves locally, and syncs to DGX server if reachable.
 */
export async function reportCampusAlert(alert: {
  location: string;
  category: 'hazard' | 'obstacle' | 'sign' | 'room_status';
  label: string;
  description: string;
  severity?: 'low' | 'medium' | 'high';
}): Promise<CampusAlert> {
  const now = new Date();
  const newAlert: CampusAlert = {
    id: Date.now(),
    location: alert.location,
    category: alert.category,
    label: alert.label,
    description: alert.description,
    severity: alert.severity || 'medium',
    reported_by: 'this_device',
    created_at: now.toISOString(),
    relative_time: 'just now',
    is_active: true,
  };

  const localAlerts = getStoredJson<CampusAlert[]>(LOCAL_STORAGE_KEY_ALERTS, SEED_ALERTS);
  const updated = [newAlert, ...localAlerts];
  setStoredJson(LOCAL_STORAGE_KEY_ALERTS, updated);

  const apiUrl = getApiUrl();
  if (apiUrl) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${apiUrl}/v1/memory/observe`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', ...getSightHeaders() },
        body: JSON.stringify({
          location: alert.location,
          category: alert.category,
          label: alert.label,
          description: alert.description,
          severity: alert.severity || 'medium',
        }),
      });
      clearTimeout(timeout);
      if (res.ok) {
        const serverItem = (await res.json()) as CampusAlert;
        return serverItem;
      }
    } catch {}
  }

  return newAlert;
}

/**
 * "What Changed?" — Compares currently detected objects against stored baseline memory.
 * Runs 100% locally on-device and can sync with server.
 */
export async function checkWhatChanged(
  location: string,
  currentDetections: LocalDetection[],
  updateBaseline = false
): Promise<WhatChangedResult> {
  const apiUrl = getApiUrl();

  // Try server first if configured
  if (apiUrl) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      const res = await fetch(`${apiUrl}/v1/memory/what-changed`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', ...getSightHeaders() },
        body: JSON.stringify({
          location,
          current_objects: currentDetections.map((d) => ({
            label: d.label,
            x: d.x,
            y: d.y,
            width: d.width,
            height: d.height,
          })),
          update_baseline: updateBaseline,
        }),
      });
      clearTimeout(timeout);
      if (res.ok) {
        return (await res.json()) as WhatChangedResult;
      }
    } catch {
      // Fallback to local on-device delta logic
    }
  }

  // 100% Offline on-device delta calculation
  const baselines = getStoredJson<Record<string, LocalDetection[]>>(
    LOCAL_STORAGE_KEY_BASELINES,
    {
      'Room 204': [
        { label: 'table', confidence: 0.9, x: 0.2, y: 0.5, width: 0.6, height: 0.4 },
        { label: 'chair', confidence: 0.85, x: 0.25, y: 0.55, width: 0.15, height: 0.25 },
      ],
      'Library Corridor': [
        { label: 'door', confidence: 0.88, x: 0.7, y: 0.2, width: 0.25, height: 0.7 },
      ],
    }
  );

  const baseline = baselines[location];
  if (!baseline || updateBaseline) {
    baselines[location] = currentDetections;
    setStoredJson(LOCAL_STORAGE_KEY_BASELINES, baselines);
    return {
      has_changed: false,
      delta_summary: `Baseline saved for ${location} with ${currentDetections.length} objects.`,
      speech: `Baseline recorded for ${location}. From now on, GuideLens will only announce what changes.`,
      added_objects: [],
      removed_objects: [],
      moved_objects: [],
    };
  }

  const baseLabels = baseline.map((b) => b.label.toLowerCase());
  const currLabels = currentDetections.map((c) => c.label.toLowerCase());

  const added: string[] = [];
  for (const label of currLabels) {
    if (baseLabels.indexOf(label) === -1 && added.indexOf(label) === -1) {
      added.push(label);
    }
  }

  const removed: string[] = [];
  for (const label of baseLabels) {
    if (currLabels.indexOf(label) === -1 && removed.indexOf(label) === -1) {
      removed.push(label);
    }
  }

  const moved: string[] = [];
  for (const curr of currentDetections) {
    const match = baseline.find((b) => b.label.toLowerCase() === curr.label.toLowerCase());
    if (match && Math.abs(curr.x - match.x) > 0.22) {
      if (moved.indexOf(curr.label) === -1) moved.push(curr.label);
    }
  }

  const hasChanged = added.length > 0 || moved.length > 0;
  if (!hasChanged) {
    return {
      has_changed: false,
      delta_summary: 'No changes detected since earlier.',
      speech: `All objects in ${location} remain in their usual positions. Your path is clear.`,
      added_objects: [],
      removed_objects: [],
      moved_objects: [],
    };
  }

  const parts: string[] = [];
  if (added.length > 0) {
    parts.push(`Something new is in your path: ${added.slice(0, 2).join(', ')}`);
  }
  if (moved.length > 0) {
    parts.push(`A ${moved.slice(0, 2).join(', ')} on your usual path has moved`);
  }

  return {
    has_changed: true,
    delta_summary: parts.join('; '),
    speech: `Notice in ${location}: ${parts.join('. ')}.`,
    added_objects: added,
    removed_objects: removed,
    moved_objects: moved,
  };
}

/**
 * Deaf-Blind Bridge message management.
 */
export async function sendBridgeMessage(
  sessionId: string,
  senderRole: 'deaf' | 'blind',
  text: string
): Promise<BridgeMessage> {
  const newMsg: BridgeMessage = {
    id: Date.now(),
    session_id: sessionId,
    sender_role: senderRole,
    text: text.trim(),
    created_at: new Date().toISOString(),
    relative_time: 'just now',
  };

  const stored = getStoredJson<BridgeMessage[]>(LOCAL_STORAGE_KEY_BRIDGE, []);
  setStoredJson(LOCAL_STORAGE_KEY_BRIDGE, [...stored, newMsg].slice(-50));

  const apiUrl = getApiUrl();
  if (apiUrl) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1500);
      const res = await fetch(`${apiUrl}/v1/bridge/message`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', ...getSightHeaders() },
        body: JSON.stringify({
          session_id: sessionId,
          sender_role: senderRole,
          text: text.trim(),
        }),
      });
      clearTimeout(timeout);
      if (res.ok) {
        return (await res.json()) as BridgeMessage;
      }
    } catch {}
  }

  return newMsg;
}

export async function getBridgeMessages(sessionId: string): Promise<BridgeMessage[]> {
  const local = getStoredJson<BridgeMessage[]>(LOCAL_STORAGE_KEY_BRIDGE, []);
  const apiUrl = getApiUrl();
  if (apiUrl) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1500);
      const res = await fetch(`${apiUrl}/v1/bridge/messages?session_id=${encodeURIComponent(sessionId)}`, {
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', ...getSightHeaders() },
      });
      clearTimeout(timeout);
      if (res.ok) {
        const msgs = (await res.json()) as BridgeMessage[];
        if (Array.isArray(msgs) && msgs.length > 0) {
          setStoredJson(LOCAL_STORAGE_KEY_BRIDGE, msgs);
          return msgs;
        }
      }
    } catch {}
  }
  return local.filter((m) => m.session_id === sessionId);
}
