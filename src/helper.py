from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
def getLLamaresponse(input_text,no_words,blog_domain):

    ### LLama2 model
    llm=CTransformers(model=r"C:\Users\Siddhant\Documents\Running-Llama2-on-CPU-Machine\model\llama-2-7b-chat.ggmlv3.q4_1.bin",
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template

    template="""
        Write a blog for {blog_domain} domain for a topic {input_text}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["blog_domain","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_domain=blog_domain,input_text=input_text,no_words=no_words))
    print(response)
    return response
