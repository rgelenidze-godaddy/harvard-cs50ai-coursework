## [Quiz 1](https://cs50.harvard.edu/ai/2024/quizzes/1/#quiz-1)

Quizzes are _optional, but encouraged_. They are a good way to test your conceptual understanding, before diving into the programming projects. Consider each question below, then reveal the answer. If you didn’t get it right, consider why you may have had that misunderstanding!

## [Question 1](https://cs50.harvard.edu/ai/2024/quizzes/1/#question-1)

**Consider these logical sentences:**

1.  _If Hermione is in the library, then Harry is in the library._
2.  _Hermione is in the library._
3.  _Ron is in the library and Ron is not in the library._
4.  _Harry is in the library._
5.  _Harry is not in the library or Hermione is in the library._
6.  _Ron is in the library or Hermione is in the library._

**Which of the following logical entailments is true?**

-   Sentence 6 entails Sentence 2
-   Sentence 1 entails Sentence 4
-   Sentence 6 entails Sentence 3
-   Sentence 2 entails Sentence 5
-   Sentence 1 entails Sentence 2
-   Sentence 5 entails Sentence 6

Click here for the answer to Question 1

Sentence 2 entails Sentence 5

## [Question 2](https://cs50.harvard.edu/ai/2024/quizzes/1/#question-2)

**There are other logical connectives that exist, other than the ones discussed in lecture. One of the most common is “Exclusive Or” (represented using the symbol ⊕). The expression `A ⊕ B` represents the sentence “A or B, but not both.” Which of the following is logically equivalent to `A ⊕ B`?**

-   `(A ∨ B) ∧ ¬ (A ∧ B)`
-   `(A ∧ B) ∨ ¬ (A ∨ B)`
-   `(A ∨ B) ∧ (A ∧ B)`
-   `(A ∨ B) ∧ ¬ (A ∨ B)`

Click here for the answer to Question 2

`(A ∨ B) ∧ ¬ (A ∧ B)`

## [Question 3](https://cs50.harvard.edu/ai/2024/quizzes/1/#question-3)

**Let propositional variable R be that “It is raining,” the variable C be that “It is cloudy,” and the variable S be that “It is sunny.” Which of the following a propositional logic representation of the sentence “If it is raining, then it is cloudy and not sunny.”?**

-   `(R → C) ∧ ¬S`
-   `R → C → ¬S`
-   `R ∧ C ∧ ¬S`
-   `R → (C ∧ ¬S)`
-   `(C ∨ ¬S) → R`

Click here for the answer to Question 3

`R → (C ∧ ¬S)`

## [Question 4](https://cs50.harvard.edu/ai/2024/quizzes/1/#question-4)

**Consider, in first-order logic, the following predicate symbols. Student(x) represents the predicate that “x is a student.” Course(x) represents the predicate that “x is a course.” Enrolled(x, y) represents the predicate that “x is enrolled in y.” Which of the following is a first-order logic translation of the sentence “There is a course that Harry and Hermione are both enrolled in.”?**

-   ∃x. Course(x) ∧ Enrolled(Harry, x) ∧ Enrolled(Hermione, x)
-   ∀x. Course(x) ∧ Enrolled(Harry, x) ∧ Enrolled(Hermione, x)
-   ∃x. Enrolled(Harry, x) ∧ ∃y. Enrolled(Hermione, y)
-   ∀x. Enrolled(Harry, x) ∧ ∀y. Enrolled(Hermione, y)
-   ∃x. Enrolled(Harry, x) ∨ Enrolled(Hermione, x)
-   ∀x. Enrolled(Harry, x) ∨ Enrolled(Hermione, x)

Click here for the answer to Question 4

∃x. Course(x) ∧ Enrolled(Harry, x) ∧ Enrolled(Hermione, x)