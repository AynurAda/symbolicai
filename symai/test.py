class JsonAInurPostProcessor(PostProcessor):
    def __call__(self, response, argument) -> Any:
      if "```" in response:
        response = response.replace("```", "")
      if "jacntination" in response:
        response = response.replace("jacntination", "justification")
      if "{" not in response:
          response = "{" + response
      if "}" not in response:
          response += "}"
      first_digit_index = re.search(r"\d", response).start()
      if response[first_digit_index+1] != ",":
        response = response[:first_digit_index+1] + "," + response[first_digit_index+1:]
      try:
        match = re.search(r'\{[^{}]*\}', response)
        if ast.literal_eval(match.group(0)):
          return response
      except:
        colon_index = response.find(":")
        signal_index = response.find("signal")
        justification_index = response.find("justification")
        if response[signal_index-1] not in ["'", "\""]:
          response = response.replace("signal", "'signal'")
        if response[justification_index-1] not in ["'", "\""]:
          response = response.replace("justification", "'justification'")
        try:
          match = re.search(r'\{[^{}]*\}', response)
          if ast.literal_eval(match.group(0)):
            return response
        except:
          right_brace_index = response.rfind("}")
          if response[right_brace_index-1] != "\"":
            response = response[:right_brace_index] + "\"" + response[right_brace_index:]
          try:
            match = re.search(r'\{[^{}]*\}', response)
            if ast.literal_eval(match.group(0)):
              return response
          except:
            if (response.find("signal") < colon_index):
              if (response[colon_index+1].isdigit() or response[colon_index+2].isdigit() or response[colon_index+3].isdigit()):
                for i in range(1, 4):  # Check positions 1, 2, and 3 after the colon
                  if response[colon_index + i].isdigit():
                      signal = response[colon_index + i].isdigit()
                      break
              temp_response = response[colon_index+1:]
              colon_index = temp_response.find(":")
              if (temp_response.find("justification") < colon_index):
                justification = temp_response[colon_index+1:]
                response = json.dumps({"signal": signal, "justification": justification})
                return response
            else:
              raise AssertionError("The model output is not feasible for the JSON schema.")