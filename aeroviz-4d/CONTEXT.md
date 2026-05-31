# AeroViz-4D Context

## UI Locale And Numeric Formatting

The application UI uses English numeric formatting regardless of the operator's
system locale. Decimal values must be displayed and entered with a period (`.`),
not a comma. Use text-based numeric controls with explicit parsing when browser
native number inputs would localize decimal separators.
