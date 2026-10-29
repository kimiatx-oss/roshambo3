import * as $3Dmol from "3dmol";

export const QUERY_CARBON = "#4b5563";
export const HIT_CARBONS = ["#e09d48", "#826cd4", "#de698d", "#4bbcac"];
export const ELEMENT_COLORS = $3Dmol.elementColors.Jmol;
// A visual van der Waals envelope, not the screening engine's Gaussian field.
export const QUERY_SURFACE: $3Dmol.SurfaceStyleSpec = {
  color: QUERY_CARBON,
  opacity: 0.3,
};

export function overlayStyle(
  representation: string,
  carbon: string,
  isQuery: boolean,
  byElement = true,
): $3Dmol.AtomStyleSpec {
  const coloring = byElement
    ? { colorscheme: { prop: "elem", map: { ...ELEMENT_COLORS, C: carbon } } }
    : { color: carbon };
  return representation === "stick" || (representation === "surface" && isQuery)
    ? { stick: { ...coloring, radius: isQuery ? 0.12 : 0.16 } }
    : { sphere: { ...coloring, scale: isQuery ? 0.65 : 0.7 } };
}

export function elementHex(element: string): string {
  return `#${ELEMENT_COLORS[element].toString(16).padStart(6, "0")}`;
}
