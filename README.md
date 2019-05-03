# Lazy Bilingual Embeddings
Position based bilingual embedding trained using only sentence aligned parallel corpora

## Method 1 Single Substitution
This method makes two passes, one over each language’s corpus. In the pass it substitutes one word based on the position from the other language and outputs a mostly monolingual sentence. This method increases the dataset size significantly as new sentences are added for each individual token in both languages. Not all translations are the same length so to accommodate this, the language with a shorter length gets the remainder of the other language appended to it and the language with a longer length randomly selects from words within the same distance from the end of the sentence.
### English to Spanish
He doesn’t like soda

No doesn’t like soda

He le like soda

He doesn’t gustan soda

He doesn’t like los refrescos


### Spanish to English
No le gustan los refrescos

He le gustan los refrescos

No doesn’t gustan los refrescos

No le like los refrescos

No le gustan soda refrescos

No le gustan los soda


## Method 2 Random Positional Substitution
This method makes a single pass over the language corpora but produces several randomly substituted versions of every sentence pair. Each token in a sentence has a 50% chance of being from either language while the length is less than or equal to the shorter sentence. After passing the shorter sentences length, the longer sentence gains a higher probability of its tokens being selected and tokens further away in the shorter sentence lose probability at each additional step of being selected. It repeats this for each sentence pair max(S1length, S2length)*2
He doesn’t gustan soda refrescos
No le gustan los refrescos
He doesn't gustan soda refrescos 
No le like soda refrescos

This reduces the bias towards words of the same language and creates a smaller training corpus.


# How to run
python create_embeddings.py --lang1 [PATH to Language 1 Corpus,default=data/Books.en-es.en]

				  --lang2 [PATH to Language 2 Corpus,default=data/Books.en-es.es]
          
				  --l1 [Language code, default=en]
          
				  --l2 [Language code, default=es]
          
				  --method [1 or 2, default=1]
          
				  --dims [# of dims, default=300]
          
				  --epochs [# of training epochs, default=5]
          
				  --window [# of tokens left and right for context, default=3]
