(*
  第一问事件层级独立复核（Wolfram Language）
  输入由“生成Wolfram事件层级载荷.mjs”从只读男胎JSON快照生成。
  本脚本只复核事件计数、重复检测离散度、样本口径和A055冲突，不拟合正式模型。
*)

If[! ValueQ[payloadText],
  If[Length[$ScriptCommandLine] < 2,
    Print["用法: wolframscript -file Wolfram事件层级复核.wl <载荷JSON>"];
    Exit[2]
  ];
  payloadText = Import[Last[$ScriptCommandLine], "Text"];
];

payload = ImportString[payloadText, "RawJSON"];
rows = payload["记录"];

parseGest[s_] := Module[{clean, parts},
  clean = StringReplace[ToString[s], {"W" -> "w", "周" -> "w", " " -> ""}];
  parts = StringSplit[StringReplace[clean, "w+" -> ","], {",", "w"}];
  7 ToExpression[parts[[1]]] +
    If[Length[parts] >= 2 && parts[[2]] =!= "", ToExpression[parts[[2]]], 0]
];

normalizeDate[v_] := Module[{digits},
  Which[
    NumericQ[v] && v >= 19000101,
      digits = IntegerDigits[Round[v], 10, 8];
      {FromDigits[digits[[1 ;; 4]]], FromDigits[digits[[5 ;; 6]]], FromDigits[digits[[7 ;; 8]]]},
    NumericQ[v],
      Take[DatePlus[{1899, 12, 30}, Round[v]], 3],
    True,
      Take[DateList[v], 3]
  ]
];

pad2[n_] := IntegerString[Round[n], 10, 2];
dateText[v_] := With[{d = normalizeDate[v]},
  ToString[d[[1]]] <> "-" <> pad2[d[[2]]] <> "-" <> pad2[d[[3]]]
];

keyBI[r_] := {r[[2]], r[[3]]};
keyBH[r_] := {r[[2]], normalizeDate[r[[4]]]};
keyBIH[r_] := {r[[2]], r[[3]], normalizeDate[r[[4]]]};
keyBIHJ[r_] := {r[[2]], r[[3]], normalizeDate[r[[4]]], parseGest[r[[5]]]};

stats[data_, key_] := Module[{groups, repeated, denominator, numerator, counts},
  groups = GatherBy[data, key];
  repeated = Select[groups, Length[#] > 1 &];
  denominator = Total[(Length[#] - 1) & /@ repeated];
  numerator = Total[((Length[#] - 1) Variance[N[#[[All, 6]]]]) & /@ repeated];
  counts = KeySort[Counts[Length /@ groups]];
  <|
    "检测记录数（条）" -> Length[data],
    "唯一事件或组数（个）" -> Length[groups],
    "多记录组数（个）" -> Length[repeated],
    "多记录组内记录数（条）" -> Total[Length /@ repeated],
    "组内Y浓度合并标准差（比例，0–1）" -> If[denominator > 0, N[Sqrt[numerator/denominator], 16], 0],
    "跨越4%阈值的组数（个）" -> Count[repeated, g_ /; Min[g[[All, 6]]] < 0.04 && Max[g[[All, 6]]] >= 0.04],
    "组大小分布" -> AssociationThread[ToString /@ Keys[counts], Values[counts]]
  |>
];

pre = Select[rows, #[[1]] < 683 &];
primary = Select[pre, 70 <= parseGest[#[[5]]] < 182 &];
sensitivity = Select[pre, 70 <= parseGest[#[[5]]] <= 175 &];
scopes = <|
  "全男胎1082条" -> rows,
  "683前682条" -> pre,
  "主参考674条" -> primary,
  "敏感性670条" -> sensitivity
|>;
keys = <|"B+I" -> keyBI, "B+H" -> keyBH, "B+I+H" -> keyBIH, "B+I+H+J" -> keyBIHJ|>;
metrics = AssociationMap[
  Function[scopeName,
    AssociationMap[
      Function[keyName, stats[scopes[scopeName], keys[keyName]]],
      Keys[keys]
    ]
  ],
  Keys[scopes]
];

biRepeated = Select[GatherBy[rows, keyBI], Length[#] > 1 &];
crossDateBI = Select[biRepeated, Length[DeleteDuplicates[normalizeDate /@ #[[All, 4]]]] > 1 &];
gestConflictBI = Select[biRepeated, Length[DeleteDuplicates[parseGest /@ #[[All, 5]]]] > 1 &];
a055 = Select[rows, #[[2]] == "A055" && #[[3]] == 3 &];

result = <|
  "计算引擎" -> "Wolfram Language远端内核",
  "内核版本" -> $Version,
  "系统标识" -> $SystemID,
  "载荷SHA256" -> Hash[payloadText, "SHA256", "HexString"],
  "载荷记录数（条）" -> Length[rows],
  "事件层级复算" -> metrics,
  "层级诊断" -> <|
    "跨检测日期的多记录B+I组数（个）" -> Length[crossDateBI],
    "孕周冲突B+I组" -> ({#[[1, 2]], #[[1, 3]]} & /@ gestConflictBI),
    "A055第3次抽血" -> (<|
      "序号" -> #[[1]],
      "检测日期" -> dateText[#[[4]]],
      "解析孕周（天）" -> parseGest[#[[5]]],
      "Y染色体浓度（比例，0–1）" -> #[[6]]
    |> & /@ a055)
  |>,
  "验收" -> <|
    "全表事件数1021_1063_1064" -> (
      metrics["全男胎1082条"]["B+I"]["唯一事件或组数（个）"] == 1021 &&
      metrics["全男胎1082条"]["B+I+H"]["唯一事件或组数（个）"] == 1063 &&
      metrics["全男胎1082条"]["B+I+H+J"]["唯一事件或组数（个）"] == 1064
    ),
    "重复组40_19_18且组内记录101_38_36" -> (
      metrics["全男胎1082条"]["B+I"]["多记录组数（个）"] == 40 &&
      metrics["全男胎1082条"]["B+I"]["多记录组内记录数（条）"] == 101 &&
      metrics["全男胎1082条"]["B+I+H"]["多记录组数（个）"] == 19 &&
      metrics["全男胎1082条"]["B+I+H"]["多记录组内记录数（条）"] == 38 &&
      metrics["全男胎1082条"]["B+I+H+J"]["多记录组数（个）"] == 18 &&
      metrics["全男胎1082条"]["B+I+H+J"]["多记录组内记录数（条）"] == 36
    ),
    "样本数674_670" -> (Length[primary] == 674 && Length[sensitivity] == 670),
    "A055冲突148_143天" -> (
      Sort[a055[[All, 1]]] == {240, 241} &&
      Sort[parseGest /@ a055[[All, 5]]] == {143, 148}
    )
  |>,
  "能力边界" -> <|
    "直接读取本地Excel" -> False,
    "证明本机Windows安装Wolfram" -> False,
    "包含正式模型拟合" -> False
  |>
|>;

jsonText = ExportString[result, "RawJSON", "Compact" -> True];
(* 当前远端接口的RawJSON导出会把UTF-8字节误显示为ISO-8859-1字符；反解后恢复中文。 *)
FromCharacterCode[ToCharacterCode[jsonText, "ISO8859-1"], "UTF8"]
