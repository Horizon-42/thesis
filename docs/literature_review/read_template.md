# First Pass
Read title, abstract, introduction, section and subsection titles, related work, and conclusion
adequate for papers aren't in your research.

## Category
attack, defense, analysis 
A measurement paper? decription of a research prototype?

## Context
**Which other papers is it related to?** If it is an attack paper, what are the other existing attacks that target the same threat model? If it is a defense paper, what are the other existing defenses that share a similar security goal as this paper?
**Which theoretical bases were used to analyze the problem?**

## Cirrectness
Do the assumptions appear to be valid?

## Contributions
What are claimed as the paper’s main contributions?

## Clarity
Is the paper well written?

# Second pass
Read the paper with greater care, but ignore details such as proofs. Look carefully at the figures, diagrams and other illustrations in the paper. The goal is to grasp the key content of the paper.

- Pay special attention to graphs. Are the axes properly labeled? Are results shown with error bars, so that conclusions are statistically significant? Common mistakes like these will separate rushed, shoddy work from the truly excellent.
- Remember to mark relevant unread references for further reading

Questions to answer:
## What are the motivations for this work?
Example:
For a security research paper, there is an expectation that the paper either discovers a new security threat or solves a security problem that no one else has published in the literature.
- To derive the motivation of the paper, you need to consider why the security problem has not been discovered or why it does not have a trivial solution.
- There is also an implication that previous attacks or solutions to the problem are inadequate. What are the previous attacks/solutions and why are they inadequate?
- Finally, the motivation and statement of the problem are distilled into a research question, the question that the paper sets out to answer.

## What is the proposed attack or defense?
- This is also called the idea of the paper. This is the proposed answer to the research question.
- There should also be an answer to the question of why it is believed that this attack/defense will work, and be better than previous work.
- There should also be a discussion about how the attack/defense is achieved (designed and implemented) or is at least achievable.

## What is the work’s evaluation of the proposed solution?
- An idea alone is usually not adequate for the publication of a research paper. The evaluation is the concrete engagement of the research question.
- What argument, implementation, and/or experiment makes the case for the value of the ideas? What benefits or problems are identified?

## What is your analysis of the identified problem, idea and evaluation?
- Is this a good idea? What flaws do you perceive in the work? What are the most interesting points made? What are the most controversial ideas or points made?
- For work that has practical implications, you also want to ask: Is this attack/defense really going to work, who can pull off the attack, and who would want the defense solution?

## What are the contributions?
The contributions in a paper may be many and varied. Beyond the insights on the research question, a few additional possibilities include: ideas, software, experimental techniques, or an area survey.

## What are future directions for this research?
Not only what future directions do the authors identify, but what ideas did you come up with while reading the paper? Sometimes these may be identified as shortcomings or other critiques in the current work.

## What questions are you left with?
What questions would you like to raise in an open discussion of the work? What do you find confusing or difficult to understand? By taking the time to list several, you will be forced to think more deeply about the work.

## What is your take-away message from this paper?
Sum up the main implication of the paper from your perspective. This is useful for very quick review to refresh your memory. It also forces you to try to identify the essence of the work.

## The third pass
- The key to the third pass is to attempt to virtually re-implement the paper: that is, making the same assumptions as the authors, re-create the work. By comparing this re-creation with the actual paper, you can easily identify not only a paper’s innovations, but also its hidden failings and assumptions. This is usually needed when you are reviewing the paper as a program committee member.
- You should identify and challenge every assumption in every statement. During this pass, you should also jot down ideas for future work.