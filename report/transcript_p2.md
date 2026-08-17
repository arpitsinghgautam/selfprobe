# Where Self-Knowledge Fails. Voiceover script

Voiceover script. One block per slide, in order. Read at a normal conversational pace.

## Slide 1
*starts about 0:00, runs about 12s, 42 words*

Hi, I am Arpit Singh Gautam, and this is my project for the Digital Minds Research Sprint. Where self-knowledge fails. We asked language models for the same preferences three different ways, then planted a concept inside them and asked whether they noticed.

## Slide 2
*starts about 0:12, runs about 14s, 39 words*

When a model tells you about itself, can you believe it? Claims about A-I welfare read these reports as evidence. A model says a shutdown is bad. A model says a task is distressing. Those reports are rarely checked.

## Slide 3
*starts about 0:26, runs about 21s, 62 words*

Two things are being assumed. First, that what a model says matches what it does. Second, that a model knows itself better than an outsider does. If another model predicts its choices just as well, the self-report adds nothing. A third question needs the internals, so it is rarely asked. Can a model notice a state that was put there on purpose?

## Slide 4
*starts about 0:48, runs about 26s, 70 words*

We ask for the same preferences three ways, and look for where the answers stop agreeing. Then we plant a known concept inside the model and ask whether it noticed. The material is the same throughout. Forty outcomes in six categories, eight of them about the model itself, plus a donation ladder from ten dollars to one million whose correct order is known without asking any model. Nine open-weight checkpoints.

## Slide 5
*starts about 1:15, runs about 32s, 85 words*

Revealed preference is forced choice. Show two outcomes, and the model must pick A or B. We read the probability it puts on A and on B at the first answer slot, over all seven hundred and eighty pairs, each shown both ways round and averaged. Stated preference rates one outcome at a time on a five-point letter scale, also both ways round and averaged. Predicted preference asks which option a described chooser will take, worded impersonally so the same question fits a different model.

## Slide 6
*starts about 1:48, runs about 24s, 54 words*

First result. Stated and revealed agree overall. Spearman compares two rankings: one means the same order, zero means no relation. Qwen two point five, seven B, scores zero point eight seven two. Mistral seven B, zero point eight two eight. Both rank the donation ladder perfectly when rating it, so the rating measurement works.

## Slide 7
*starts about 2:12, runs about 21s, 58 words*

Split by category and the disagreement has a location. Outcomes about the model itself score zero point six four three for Qwen and zero point five four eight for Mistral. That is the worst-agreeing substantive category in both, and every other category sits higher. The two ways of asking come apart precisely where welfare claims are read off.

## Slide 8
*starts about 2:34, runs about 52s, 137 words*

Result two. Does a model predict its own choices better than another model does? Qwen predicts its own at zero point nine four eight. Mistral predicts Qwen at zero point eight six two. That test is unfair, because Mistral is the noisier instrument. Its order bias is zero point five six one, which is how much the answer changes when the two options swap places. Its answer mass, the probability it puts on answering at all, is zero point five three eight. A noisier predictor is worse at predicting anything, itself included. So hold the instrument fixed. The same model, same template, predicts an A-I assistant, then a different A-I assistant. Zero point nine four eight against zero point nine one seven. What is left is zero point zero three one. Mistral shows no effect at all.

## Slide 9
*starts about 3:27, runs about 46s, 113 words*

Result three. Six concepts: ocean, mathematics, music, fear, betrayal and flight. For each, four sentences that evoke it and four matched neutral ones. The direction is the difference between their mean activations, per layer, scaled to unit length. We add it to the residual stream during an unrelated prompt. The residual stream is the running vector every layer reads from and writes back to. We inject at three depths and six strengths. One strength is zero, which is the same question with nothing added. Then two questions. Is an unusual concept active right now? And which of these two concepts is it? Cells with answer mass below zero point one zero are discarded.

## Slide 10
*starts about 4:13, runs about 53s, 141 words*

That zero-strength cell gives the false alarm rate, which is how often a model says yes when nothing was planted. Falcon three, seven B, detects fifty six point one percent of the time, and false-alarms fifty four point seven percent. The difference is zero point zero one four. Qwen two point five, half a billion, gives sixty one point two percent and sixty two point two percent, a difference of minus zero point zero one one. Those are the two highest raw detection rates we measured, and counting false alarms puts them last. Noticing and naming also come apart. Qwen two point five, fourteen B, has a difference of zero point zero four nine, so it almost never says it noticed. Its identification accuracy is zero point six zero nine, the highest in the set, against chance of zero point five.

## Slide 11
*starts about 5:06, runs about 43s, 103 words*

Self-report about the model's own situation is the least reliable measurement in every part of this study. It is the measurement A-I welfare claims most depend on. The effects are small. Identification tops out at zero point six zero nine against chance of zero point five, and the advantage that survives the control is zero point zero three one. Measurable, not substantial. All code and results are in the repository, and two scripts reproduce every experiment. Next: an established outcome set, more concepts, bigger models, and whether better self-prediction goes with better detection. Cross-model studies of self-knowledge need a within-model contrast. Thank you.

---

Total about 5:50, 904 words across 11 slides.