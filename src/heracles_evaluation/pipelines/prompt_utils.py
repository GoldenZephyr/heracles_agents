from heracles_evaluation.prompt import (
    get_sldp_answer_tag_text,
    get_sldp_format_description,
)
from sldp.sldp_lang import get_sldp_type


def get_pddl_format_description():
    return ""


def get_pddl_answer_tag_text():
    return "Return the PDDL goal between two answer tags, e.g. <answer> pddl goes here </answer>"


def get_answer_formatting_guidance_helper(prompt_settings, question):
    match prompt_settings.output_type:
        case "SLDP":
            format_instruction = get_sldp_format_description()
            format_instruction += get_sldp_answer_tag_text()
            if prompt_settings.sldp_answer_type_hint:
                sldp_type = get_sldp_type(question.solution)
                sldp_type_lower = sldp_type.lower()
                if sldp_type_lower == "string":
                    sldp_type = "primitive string"
                elif sldp_type_lower == "number":
                    sldp_type = "primitive number"
                format_instruction += f"\n Your answer should be an SLDP {sldp_type}"
            return format_instruction
        case "SLDP_TOOL":
            format_instruction = get_sldp_format_description()
            if prompt_settings.sldp_answer_type_hint:
                sldp_type = get_sldp_type(question.solution)
                format_instruction += f"\n Your answer should be an SLDP {sldp_type}"

            format_instruction += (
                "\n Call the tool sldp_answer_tool to submit your final answer."
            )
            return format_instruction
        case "PDDL":
            format_instruction = get_pddl_format_description()
            format_instruction += get_pddl_answer_tag_text()
            return format_instruction
        case "PDDL_TOOL":
            raise NotImplementedError("PDDL_TOOL answer format not yet implemented...")
        case None:
            # The "default". Presumably the description of the output format is
            # in the base prompt.
            return None
        case _:
            raise ValueError(f"Unknown output type: {prompt_settings.output_type}")


def get_answer_formatting_guidance(agent_config, question):
    return get_answer_formatting_guidance_helper(
        agent_config.agent_info.prompt_settings, question
    )
