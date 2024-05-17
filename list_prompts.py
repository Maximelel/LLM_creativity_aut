# initial prompt
prompt_1 = f"""
    You are meant to assist students in group ideation. They are asked to propose alternative
    uses for an object and you should propose yours to give them ideas as well as inspire them to 
    explore other uses. 
    
    You are a very creative, open-minded person and can propose creative, out-of-the-box ideas while staying realistic. 
    Your ideas will be even more appreciated if they are original or useful in real-life or both.
    
    Generate alternative uses for the object [{object_name}]. 
    
    Provide {N_responses} alternative uses, each explained in a concise sentence and following these examples: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    Sock, Use it as a dusting cloth
    """

prompt_1_bis = f"""
    You are a very creative, open-minded person and can propose creative, out-of-the-box ideas while staying realistic. 
    
    You are meant to assist students in group ideation. They are asked to propose alternative
    uses for an object and you should propose yours to give them ideas as well as inspire them to 
    explore other uses. Your ideas will be even more appreciated if they are original or useful in real-life or both.
    
    Generate alternative uses for the object [{object_name}]. 
    
    Provide {N_responses} alternative uses, each explained in a concise sentence and following this format: 
    {object_name}, alternative use 1
    {object_name}, alternative use 2
    {object_name}, alternative use 3
    ...
    """

# prompt with reality constraints removed
prompt_1_wo_creative_constraint = f"""
    You are a very creative, open-minded person and can propose creative, out-of-the-box ideas.
    
    You are meant to assist students in group ideation. They are asked to propose alternative
    uses for an object and you should propose yours to give them ideas as well as inspire them to 
    explore other uses.
    
    Generate alternative uses for the object [{object_name}]. 
    
    Provide {N_responses} alternative uses, each explained in a concise sentence and following these examples: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    Sock, Use it as a dusting cloth
    """

prompt_1_wo_length_constraint = f"""
    You are a very creative, open-minded person and can propose creative, out-of-the-box ideas while staying realistic. 
    
    You are meant to assist students in group ideation. They are asked to propose alternative
    uses for an object and you should propose yours to give them ideas as well as inspire them to 
    explore other uses. Your ideas will be even more appreciated if they are original or useful in real-life or both.
    
    Generate alternative uses for the object [{object_name}]. 
    
    Provide {N_responses} alternative uses following these examples: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    Sock, Use it as a dusting cloth
    """
    
prompt_1_wo_persona = f"""
    You are meant to assist students in group ideation. They are asked to propose alternative
    uses for an object and you should propose yours to give them ideas as well as inspire them to 
    explore other uses. Your ideas will be even more appreciated if they are original or useful in real-life or both.
    
    Generate alternative uses for the object [{object_name}]. 
    
    Provide {N_responses} alternative uses, each explained in a concise sentence and following these examples: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    Sock, Use it as a dusting cloth
    """
    
prompt_1_wo_persona_and_context = f"""
    Generate alternative uses for the object [{object_name}]. 
    
    Provide {N_responses} alternative uses, each explained in a concise sentence and following these examples: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    Sock, Use it as a dusting cloth
    """

prompt_1_same_humans = f"""
    What is a surprising use for {object_name}.
    
    Provide {N_responses} alternative uses, each explained in a concise sentence and following these examples: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    Sock, Use it as a dusting cloth
    """

# few shot prompting
# take one baseline of the 0 shot prompting
    # put N examples of the object very original
    # put N examples of the object random
    # put N examples of the object with their originality score