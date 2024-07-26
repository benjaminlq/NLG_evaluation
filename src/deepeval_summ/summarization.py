from typing import List, Union
import asyncio

from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    print_verbose_logs,
    validate_conversational_test_case,
    trimAndLoadJson,
    check_llm_test_case_params,
)
from deepeval.metrics.summarization.template import SummarizationTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.summarization.summarization import SummarizationAlignmentVerdict, SummarizationCoverageVerdict, ScoreType
from deepeval.metrics.summarization.schema import Verdict

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
]

class CustomSummarizationMetric(SummarizationMetric):

    async def _a_generate_alignment_verdicts(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> List[SummarizationAlignmentVerdict]:

        if len(self.claims) == 0:
            return []

        verdicts: List[SummarizationAlignmentVerdict] = []
        prompt = SummarizationTemplate.generate_alignment_verdicts(
            summary_claims=self.claims, orignal_text=test_case.input
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            verdicts = [
                SummarizationAlignmentVerdict(**item)
                for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdict = await self.model.a_generate(
                    prompt, schema=Verdict
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    SummarizationAlignmentVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _generate_alignment_verdicts(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> List[SummarizationAlignmentVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[SummarizationAlignmentVerdict] = []
        prompt = SummarizationTemplate.generate_alignment_verdicts(
            summary_claims=self.claims, orignal_text=test_case.input
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            verdicts = [
                SummarizationAlignmentVerdict(**item)
                for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdict = self.model.generate(prompt, schema=Verdict)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    SummarizationAlignmentVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def measure(
        self, test_case: Union[LLMTestCase, ConversationalTestCase]
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = validate_conversational_test_case(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.claims: List[str] = self._generate_claims(test_case.actual_output)

                self.coverage_verdicts: List[SummarizationCoverageVerdict] = (
                    self._generate_coverage_verdicts(test_case)
                )
                self.alignment_verdicts: List[SummarizationAlignmentVerdict] = (
                    self._generate_alignment_verdicts(test_case)
                )
                alignment_score = self._calculate_score(ScoreType.ALIGNMENT)
                coverage_score = self._calculate_score(ScoreType.COVERAGE)
                self.score_breakdown = {
                    ScoreType.ALIGNMENT.value: alignment_score,
                    ScoreType.COVERAGE.value: coverage_score,
                }
                self.score = min(alignment_score, coverage_score)
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold
                if self.verbose_mode:
                    print_verbose_logs(
                        self.__name__,
                        steps=[
                            f"Claims:\n{prettify_list(self.claims)}\n",
                            f"Assessment Questions:\n{prettify_list(self.assessment_questions)}\n",
                            f"Coverage Verdicts:\n{prettify_list(self.coverage_verdicts)}\n",
                            f"Alignment Verdicts:\n{prettify_list(self.alignment_verdicts)}\n",
                            f"Score: {self.score}\nReason: {self.reason}",
                        ],
                    )
                return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True,
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = validate_conversational_test_case(test_case, self)
        check_llm_test_case_params(test_case, required_params, self)

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            self.claims = await asyncio.gather(
                self._a_generate_claims(test_case.actual_output),
            )
            (
                self.coverage_verdicts,
                self.alignment_verdicts,
            ) = await asyncio.gather(
                self._a_generate_coverage_verdicts(test_case),
                self._a_generate_alignment_verdicts(test_case),
            )
            alignment_score = self._calculate_score(ScoreType.ALIGNMENT)
            coverage_score = self._calculate_score(ScoreType.COVERAGE)
            self.score_breakdown = {
                ScoreType.ALIGNMENT.value: alignment_score,
                ScoreType.COVERAGE.value: coverage_score,
            }
            self.score = min(alignment_score, coverage_score)
            self.reason = await self._a_generate_reason()
            self.success = self.score >= self.threshold
            if self.verbose_mode:
                print_verbose_logs(
                    self.__name__,
                    steps=[
                        f"Claims:\n{prettify_list(self.claims)}\n",
                        f"Assessment Questions:\n{prettify_list(self.assessment_questions)}\n",
                        f"Coverage Verdicts:\n{prettify_list(self.coverage_verdicts)}\n",
                        f"Alignment Verdicts:\n{prettify_list(self.alignment_verdicts)}\n",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )
            return self.score