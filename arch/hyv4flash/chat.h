/* Hy4-flash's reply format, from the release's chat_template.jinja: every marker carries the
 * ":opensource" suffix (the template's HYTK), and the ids are in the release's tokenizer.json:
 *
 *   <think:opensource> 120029   </think:opensource> 120030
 *   <tool_calls:opensource> 120035 </tool_calls:opensource> 120036
 *   <tool_call:opensource> 120033  </tool_call:opensource> 120034
 *   <arg_key:opensource> 120040  </arg_key:opensource> 120041
 *   <arg_value:opensource> 120042  </arg_value:opensource> 120043
 *
 * A reply is  [<think>REASONING</think>]CONTENT[<tool_calls><tool_call>NAME
 *   <arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>...</tool_calls>]  and ends with
 * <｜hy_end:opensource｜> (120025, the generation config's eos). The template writes the name
 * straight after the call opener, with no delimiter and no newlines anywhere, and an argument
 * that is not a string as JSON (`tojson`), so a string-typed value is taken raw and the rest as
 * JSON. The think tags are ordinary (non-special) tokens, so they decode as text; the generation
 * prompt ends in `<think>` (reasoning_effort high) or `<think></think>` (no_think), which the
 * parser reads off the rendered prompt. */
#pragma once
#include "rad_builder.h"

namespace hyv4flash {

inline const RadChatFormat* hy4_chat_format() {
    static const char* const breaks[] = {"<tool_calls:opensource>"};
    static const char* const triggers[] = {"<tool_calls:opensource>"};
    static const RadChatFormat f = [] {
        RadChatFormat x{};
        x.struct_size = sizeof(x);
        x.name = "plugin:hyv4flash";
        x.reasoning_open = "<think:opensource>";
        x.reasoning_close = "</think:opensource>";
        x.reasoning_breaks = breaks;
        x.n_reasoning_breaks = 1;
        x.reasoning_in_prompt = RAD_CHAT_THINK_FROM_PROMPT;
        x.section_open = "<tool_calls:opensource>";
        x.section_close = "</tool_calls:opensource>";
        /* The name has no delimiter of its own: it ends where the first argument, or the call's
         * end, begins. */
        x.name_prefix = "<tool_call:opensource>";
        x.call_end = "</tool_call:opensource>";
        x.arg_prefix = "<arg_key:opensource>";
        x.arg_name_suffix = "</arg_key:opensource>";
        x.arg_value_prefix = "<arg_value:opensource>";
        x.arg_value_suffix = "</arg_value:opensource>";
        x.string_value = RAD_CHAT_VALUE_RAW;
        x.other_value = RAD_CHAT_VALUE_JSON;
        x.triggers = triggers;
        x.n_triggers = 1;
        return x;
    }();
    return &f;
}

/* The plugin's `rad_arch_chat_format` (RAD_ARCH_CHAT_FORMAT(hyv4flash)). */
inline const RadChatFormat* chat_format(const RadModelMeta*) { return hy4_chat_format(); }

}  /* namespace hyv4flash */
