import re
from typing import Dict, List, Optional, Union
import os
import json
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

def get_llm_output(
    prompt: str,
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    json_object: bool = False,
    return_raw: bool = False,
) -> str:
    """
    Get output from LLM models.
    
    Args:
        prompt: Input prompt text
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        stop: Stop sequences
        json_object: Whether to return JSON object
        return_raw: Whether to return raw response
        
    Returns:
        Generated text from LLM
    """
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _call_llm_api():
        if "gpt" in model:
            # Set API key from environment
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            messages = [{"role": "user", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                response_format={"type": "json_object"} if json_object else None
            )
            return response
        else:
            raise ValueError(f"Unsupported model: {model}")

    try:
        response = _call_llm_api()
        
        if return_raw:
            return response
            
        if "gpt" in model:
            return response.choices[0].message.content
            
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        raise

def extract_code_from_actions(actions: str) -> Dict:
    """
    Extract code components from actions string.
    
    Args:
        actions: String containing the actions/code
        
    Returns:
        Dict containing extracted code components
    """
    # Extract parameter dict
    param_dict_match = re.search(r'parameter_dict\s*=\s*{([^}]+)}', actions)
    param_dict = param_dict_match.group(1) if param_dict_match else ''
    
    # Extract main function
    func_match = re.search(r'def get_node_score_dict[^}]+}', actions)
    main_func = func_match.group(0) if func_match else ''
    
    # Extract helper functions
    helper_funcs = []
    helper_matches = re.finditer(r'def (?!get_node_score_dict)[^\n]+\n(?:[ \t]+[^\n]+\n)+', actions)
    for match in helper_matches:
        helper_funcs.append(match.group(0))
    
    return {
        'parameter_dict': param_dict,
        'main_function': main_func,
        'helper_functions': helper_funcs
    }

def merge_code_components(components: List[Dict]) -> str:
    """
    Merge code components into a single actions string.
    
    Args:
        components: List of code component dictionaries
        
    Returns:
        Merged actions string
    """
    # Combine helper functions
    helper_funcs = set()
    for comp in components:
        helper_funcs.update(comp.get('helper_functions', []))
    
    # Get main function from best performing component
    main_func = components[0].get('main_function', '')
    
    # Merge parameter dicts
    param_dicts = [comp.get('parameter_dict', '') for comp in components]
    merged_params = merge_parameter_dicts(param_dicts)
    
    # Combine everything
    merged_code = []
    merged_code.extend(helper_funcs)
    merged_code.append(f"parameter_dict = {{{merged_params}}}")
    merged_code.append(main_func)
    
    return '\n\n'.join(merged_code)

def merge_parameter_dicts(param_dicts: List[str]) -> str:
    """
    Merge parameter dictionary strings.
    
    Args:
        param_dicts: List of parameter dictionary strings
        
    Returns:
        Merged parameter dictionary string
    """
    # Parse parameter strings into actual dicts
    parsed_dicts = []
    for param_str in param_dicts:
        try:
            param_dict = eval(f"{{{param_str}}}")
            parsed_dicts.append(param_dict)
        except:
            continue
            
    # Merge dictionaries
    merged = {}
    for d in parsed_dicts:
        for k, v in d.items():
            if k not in merged:
                merged[k] = v
            else:
                # Average numerical values
                if isinstance(v, (int, float)) and isinstance(merged[k], (int, float)):
                    merged[k] = (merged[k] + v) / 2
                    
    # Convert back to string format
    return ', '.join(f"'{k}': {v}" for k, v in merged.items()) 