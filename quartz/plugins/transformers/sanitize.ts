import { QuartzTransformerPlugin } from "../types"

export const SanitizeText: QuartzTransformerPlugin = () => {
    return {
      name: "SanitizeText",
      textTransform(_ctx, src) {
        src = src.toString()
        src = src.replaceAll("??", "");
        return src;
      },
    }
  }