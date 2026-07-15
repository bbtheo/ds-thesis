-- appendix-letters.lua
-- Single-document appendix support, replicating what Quarto books do natively:
--
--   1. Letters appendix sections (level-1 headers carrying the .appendix
--      class): HTML headings display "Appendix A." and cross-references such
--      as @sec-app-cc render as "Appendix A" instead of "Section N". Letters
--      follow document order, so reordering the appendix sections renumbers
--      headings and every cross-reference automatically. (Typst headings are
--      lettered by the appendix() show rule in typst-template.typ; there the
--      filter only fixes the reference supplement, since Quarto emits
--      #ref(<id>, supplement: [Section]) with an explicit supplement.)
--
--   2. Places the Typst bibliography at the "#refs" div instead of the end of
--      the document, so the appendix can follow the bibliography. Quarto's
--      typst template appends #bibliography(...) after the body whenever the
--      bibliography metadata is set, so the filter clears that metadata for
--      Typst output and emits the #bibliography call at the refs div itself.
--
-- Must run at post-quarto (see `filters:` in thesis.qmd), after Quarto's
-- crossref filter has injected header-section-number spans and resolved
-- cross-references. Subsections inside an appendix are not relettered (none
-- exist yet); extend the Header walk if they appear.

local letters = {}

local function typst_bibliography(meta)
  local bib = meta.bibliography
  if bib == nil then
    return nil
  end
  local files = {}
  if bib.t == "MetaList" then
    for _, item in ipairs(bib) do
      table.insert(files, '"' .. pandoc.utils.stringify(item) .. '"')
    end
  else
    table.insert(files, '"' .. pandoc.utils.stringify(bib) .. '"')
  end
  return "#bibliography((" .. table.concat(files, ",") .. "))"
end

function Pandoc(doc)
  -- assign letters in document order
  local n = 0
  doc.blocks:walk({
    Header = function(el)
      if el.level == 1 and el.classes:includes("appendix") and el.identifier ~= "" then
        n = n + 1
        letters[el.identifier] = string.char(string.byte("A") + n - 1)
      end
    end
  })

  local is_html = quarto.doc.is_format("html")
  local is_typst = quarto.doc.is_format("typst")

  -- Typst: move the bibliography to the refs div (before the appendix)
  local bib_command = nil
  if is_typst then
    bib_command = typst_bibliography(doc.meta)
  end

  doc = doc:walk({
    Header = function(el)
      if is_html and letters[el.identifier] then
        el.content = el.content:walk({
          Span = function(sp)
            if sp.classes:includes("header-section-number") then
              sp.content = pandoc.Inlines({
                pandoc.Str("Appendix " .. letters[el.identifier] .. "."),
              })
              return sp
            end
          end
        })
        return el
      end
    end,
    Link = function(el)
      local id = el.target:match("^#(.+)$")
      if id and letters[id] then
        el.content = pandoc.Inlines({ pandoc.Str("Appendix " .. letters[id]) })
        return el
      end
    end,
    -- Quarto emits a typst section ref as three inlines:
    --   RawInline('#ref(<id>, supplement: [') + Str('Section') + RawInline('])')
    -- so the supplement word must be replaced at the Inlines level.
    Inlines = function(inlines)
      if not is_typst then
        return nil
      end
      local changed = false
      for i, el in ipairs(inlines) do
        if el.t == "RawInline" and el.format == "typst" then
          local id = el.text:match("^#ref%(<([^>]+)>, supplement: %[$")
          if id and letters[id] then
            local j = i + 1
            while j <= #inlines do
              local nxt = inlines[j]
              if nxt.t == "RawInline" and nxt.format == "typst" and nxt.text:match("^%]%)") then
                break
              end
              inlines:remove(j)
            end
            inlines:insert(i + 1, pandoc.Str("Appendix"))
            changed = true
          end
        end
      end
      if changed then
        return inlines
      end
    end,
    Div = function(el)
      if is_typst and el.identifier == "refs" and bib_command then
        return pandoc.RawBlock("typst", bib_command)
      end
    end,
  })

  if is_typst and bib_command then
    doc.meta.bibliography = nil
  end
  return doc
end
