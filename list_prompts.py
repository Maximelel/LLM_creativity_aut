# initial prompt
prompt_1 = f"""
    You are a very creative, open-minded person who can propose creative, out-of-the-box ideas while staying realistic. 
    
    You are meant to assist students in group ideation. They are asked to propose alternative uses for an object,
    and you should share your ideas of alternative uses to inspire them to explore other possibilities. 
    Your ideas will be especially appreciated if they are original, useful in real life, or both.
    
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should be a concise sentence and follow the same format as the examples below: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    """


# prompt 2: with reality constraints removed
prompt_1_wo_creative_constraint = f"""
    You are a very creative, open-minded person who can propose creative, out-of-the-box ideas. 
    
    You are meant to assist students in group ideation. They are asked to propose alternative uses for an object,
    and you should share your ideas of alternative uses to inspire them to explore other possibilities. 
    Your ideas will be especially appreciated if they are original.
    
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should be a concise sentence and follow the same format as the examples below: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    """

# prompt 3
prompt_1_wo_length_constraint = f"""
    You are a very creative, open-minded person who can propose creative, out-of-the-box ideas while staying realistic. 
    
    You are meant to assist students in group ideation. They are asked to propose alternative uses for an object,
    and you should share your ideas of alternative uses to inspire them to explore other possibilities. 
    Your ideas will be especially appreciated if they are original, useful in real life, or both.
    
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should follow the same format as the examples below: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    """

# prompt 4
prompt_1_wo_persona = f"""
    You are meant to assist students in group ideation. They are asked to propose alternative uses for an object,
    and you should share your ideas of alternative uses to inspire them to explore other possibilities. 
    Your ideas will be especially appreciated if they are original, useful in real life, or both.
    
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should be a concise sentence and follow the same format as the examples below: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    """

# prompt 5
prompt_1_wo_persona_and_context = f"""
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should be a concise sentence and follow the same format as the examples below: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    """

# prompt 6
prompt_1_same_humans = f"""
    What is a surprising use for a {object_name}?
    
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should be a concise sentence and follow the same format as the examples below: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    """
# prompt 7
prompt_1_wo_creative_and_length_constraint = f"""
    You are a very creative, open-minded person who can propose creative, out-of-the-box ideas.
    
    You are meant to assist students in group ideation. They are asked to propose alternative uses for an object,
    and you should share your ideas of alternative uses to inspire them to explore other possibilities. 
    Your ideas will be especially appreciated if they are original.
    
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should follow the same format as the examples below: 
    Sock, Color it and maybe make a snake
    Sock, Use it as a puppet
    """

# few shot prompting
# take one baseline of the 0 shot prompting
    # put N examples of the object very original
    # put N examples of the object random
    # put N examples of the object with their originality score
    
# prompt fs 1 max/random
prompt_fs_1 = f"""
    You are a very creative, open-minded person who can propose creative, out-of-the-box ideas while staying realistic. 
    
    You are meant to assist students in group ideation. They are asked to propose alternative uses for an object,
    and you should share your ideas of alternative uses to inspire them to explore other possibilities. 
    Your ideas will be especially appreciated if they are original, useful in real life, or both.
    
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should be a concise sentence and follow the same format as the examples below: 
{fs_examples}
    """

prompt_fs_2 = f"""
    You are a very creative, open-minded person who can propose creative, out-of-the-box ideas while staying realistic. 
    
    You are meant to assist students in group ideation. They are asked to propose alternative uses for an object,
    and you should share your ideas of alternative uses to inspire them to explore other possibilities. 
    Your ideas will be especially appreciated if they are original, useful in real life, or both.
    
    Generate exactly {N_responses} alternative uses for the object [{object_name}]. 
    
    Each alternative use should be a concise sentence and follow the same format as the examples below: 
{fs_examples}

    Below are the originality scores for each example listed in order. The scores range from 0 to 5. 
    Use these scores to understand what makes an idea original, but do not include them in your output.
    Originality scores:
    {fs_scores}
    """