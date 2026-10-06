/**
 * TypeCode.tsx
 * ------------
 * An ICAO type code, with its plain name as its tooltip where the app has one (`utils/aircraftTypeNames.ts`); a code without a
 * name is plain text, and an untyped aircraft is a dash.
 */

export default function TypeCode({ code, name }: { code: string | null; name?: string }) {
  if (code === null) return <>—</>;
  return name === undefined ? <>{code}</> : <span className="traffic-job-type" title={name}>{code}</span>;
}
