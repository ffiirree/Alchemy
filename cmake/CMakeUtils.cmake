
# log
function(log text)
    set(status_cond)

    foreach(arg ${ARGN})
        list(APPEND status_cond ${arg})
    endforeach()

    set(status_placeholder_length 32)
    string(RANDOM LENGTH ${status_placeholder_length} ALPHABET " " status_placeholder)
    string(LENGTH "${text}" status_text_length)
    if(status_text_length LESS status_placeholder_length)
          string(SUBSTRING "${text}${status_placeholder}" 0 ${status_placeholder_length} status_text)
    else()
          set(status_text "${text}")
    endif()

    string(REPLACE ";" " " status_cond "${status_cond}")
      string(REGEX REPLACE "^[ \t]+" "" status_cond "${status_cond}")
      message(STATUS "~ ${status_text} ${status_cond}")
endfunction()