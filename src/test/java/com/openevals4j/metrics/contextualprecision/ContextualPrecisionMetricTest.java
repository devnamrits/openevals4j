package com.openevals4j.metrics.contextualprecision;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.openevals4j.metrics.models.EvaluationContext;
import com.openevals4j.metrics.models.EvaluationResult;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.List;

import dev.langchain4j.model.output.Response;
import lombok.SneakyThrows;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class ContextualPrecisionMetricTest {

  private ContextualPrecisionMetric contextualPrecisionMetric;
  private ChatLanguageModel mockClient;

  @BeforeEach
  void setUp() {
    mockClient = mock(ChatLanguageModel.class);
    contextualPrecisionMetric =
            spy(ContextualPrecisionMetric.builder()
                    .evaluatorLLM(mockClient)
                    .objectMapper(new ObjectMapper())
                    .build());
  }

  // Helper to create mocked Response that returns the given text on content().text()
  private Response<AiMessage> makeMockResponse(String text) {
    @SuppressWarnings("unchecked")
    Response<AiMessage> res = mock(Response.class);
    AiMessage ai = mock(AiMessage.class);
    // adapt method names if different
    when(res.content()).thenReturn(ai);
    when(ai.text()).thenReturn(text);
    return res;
  }

  @Test
  void evaluate() {
    Response<AiMessage> verdictPromptResponse = makeMockResponse("```json\n[\n    {\n        \"verdict\": \"yes\",\n        \"reason\": \"The context directly provides the information needed for the expected output, stating that 'All customers are eligible for a 30 day full refund at no extra cost', which directly answers the question about what happens if the shoes don't fit.\"\n    }\n]\n```");
    Response<AiMessage> reasonPromptResponse = makeMockResponse("```json\n{\n    \"reason\": \"The score is 1.0 because all relevant nodes are perfectly ranked! The first node in the retrieval contexts is highly relevant, explaining that 'The context 'All customers are eligible for a 30 day full refund at no extra cost' directly provides the information needed to form the expected output, which is a verbatim match.'\"\n}\n```");
    // Return verdict on first generate(...) call, reason on second
    when(mockClient.generate(any(UserMessage.class)))
            .thenReturn(verdictPromptResponse)
            .thenReturn(reasonPromptResponse);
    doReturn(mockClient).when(contextualPrecisionMetric).getEvaluatorLLM();
    EvaluationResult evaluationResult =
            contextualPrecisionMetric.evaluate(
                    EvaluationContext.builder()
                            .userInput("What if these shoes don't fit?")
                            .actualResponse("We offer a 30-day full refund at no extra cost.")
                            .expectedResponse("You are eligible for a 30 day full refund at no extra cost")
                            .retrievedContexts(
                                    List.of(
                                            "All customers are eligible for a 30 day full refund at no extra cost."))
                            .build());

    Assertions.assertEquals(1.0, evaluationResult.getScore());
  }

  @Test
  @SneakyThrows
  void whenLLMReturnsNonJson_thenMethodThrowsJsonProcessingException() {
    Response<AiMessage> malformedResponse = makeMockResponse("This is not JSON");

    when(mockClient.generate(any(ChatMessage.class))).thenReturn(malformedResponse);

    doReturn(mockClient).when(contextualPrecisionMetric).getEvaluatorLLM();

    Method m = contextualPrecisionMetric.getClass().getDeclaredMethod("generateVerdicts", String.class, String.class, List.class);
    m.setAccessible(true);

    InvocationTargetException thrown =
            assertThrows(InvocationTargetException.class, () -> m.invoke(contextualPrecisionMetric, "inp", "exp", Collections.emptyList()));

    // The underlying cause should be a JsonProcessingException (the ObjectMapper failed to parse)
    Throwable cause = thrown.getCause();
    assertNotNull(cause, "Expected a cause");
    assertInstanceOf(JsonProcessingException.class, cause, "Expected JsonProcessingException as the cause but got: " + cause.getClass());
  }
}
