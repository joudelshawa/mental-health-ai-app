import twint
import nest_asyncio
#if not running run - !pip3 install --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint
#for me it is working on the google colab

####For look up
import twint

c = twint.Config()
c.Username = "RobertoEntio"
c.Since = "2019-05-01"
c.Until = "2019-05-10"

twint.run.Lookup(c)
#################################
######Get as DataFramegit
import pandas as pd
import twint
import nest_asyncio
nest_asyncio.apply()
c = twint.Config()
c.Search = "RobertoEntio"
c.Limit = 1
c.Pandas = True
c.Hide_output= True
twint.run.Search(c)
twint.storage.panda.Tweets_df
