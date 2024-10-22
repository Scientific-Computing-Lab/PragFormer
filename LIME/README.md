In the example above we present representative examples and the corresponding prediction of
PragFormer over the benchmark tests. Explaining and understanding the reason behind a model's prediction is a difficult task.
Nonetheless, there are many algorithms that attempt to give an explanation or an intuition for classifiers' decisions, such as [LIME](https://github.com/marcotcr/lime). LIME studies the connections between keywords (tokens) 
of an input and the change in the prediction of a model once they are removed. Finally, LIME presents 
the probability that the keyword affected the prediction. In our case, this might indicate how PragFormer
focuses on keywords and statements. Thus, in order to gain an intuition for the predicted outcome of PragFormer,
we applied LIME on the four examples

The first example presents a code snippet taken from PolyBench, in which PragFormer managed to 
identify correctly the need for an OpenMP directive. In this example, LIME pinpoints the variables _j_, 
_POLYBENCH\_LOOP\_BOUND_, _A_ and _y_1_ as the main contributors to the decision of the model. 
This indicates that PragFormer focuses on the loop variable and the arrays, as it should. The second example 
presents a snippet taken from PolyBench and contains an I/O operation, thus, there is no OpenMP directive. 
LIME identifies the keyword _fprintf_ and _stderr_ as the reason why the model classified the snippet without OpenMP. To verify this claim, we removed these two keywords, and PragFormer predicts an OpenMP directive. This probably indicates that PragFormer understands that these specific keywords are why there is no need for an OpenMP directive. The third example, taken from SPEC-OMP, contains an OpenMP directive that PragFormer fails to predict correctly. By observing LIME's result, the two variables, _ssize\_t_ and _IndexPacket_, are the main reason for its incorrect prediction. After removing both variables, PragFormer predicts an OpenMP directive. This might be due to the model's unfamiliarity with these keywords and how they affect the code. The fourth example, taken from PolyBench, contains an assignment into three arrays. While PragFormer predicts (correctly) that there should be an OpenMP directive, the example does not. As with the first example, LIME pinpoints the loop-variable _j_ and the arrays as the reason for its prediction. This further indicates that the model does focus on the correct variables while making its decision.

In summary, it appears most likely that PragFormer focuses on the core variables and statements in order to predict its directive. However, 
there are still cases in which it fails to predict correctly, possibly due to unfamiliar keywords or statements.
