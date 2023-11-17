---

excalidraw-plugin: parsed
tags: [excalidraw]

---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠==


# Text Elements
Set of target actions for all agents at time t=1 ^qduYRtMJ

Product over all timesteps ^fFaV0kJ3

Product over all agents ^XlIc92UV

Probability of agent states at time t given
all previous agent states and scene features S ^btfe74IM

Probability of all agent states at
time t=1,2,3,...T given S ^JIfiy6ng

They treat agent states as conditionally
independent at time t given the previous actions
and scene context so these two probabilities
are the same.
 ^AHiD2Dz2


# Embedded files
f00f945039192360a1dcce6c8fd10025bb190243: $$\begin{gathered} p_\theta\left(A_1, A_2, \ldots A_T \mid S\right)=\prod_{t=1}^T p_\theta\left(A_t \mid A_{<t}, S\right) \\ p_\theta\left(A_t \mid A_{<t}, S\right)=\prod_{n=1}^N p_\theta\left(a_t^n \mid A_{<t}, S\right) . \end{gathered}$$

%%
# Drawing
```json
{
	"type": "excalidraw",
	"version": 2,
	"source": "https://github.com/zsviczian/obsidian-excalidraw-plugin/releases/tag/1.9.6",
	"elements": [
		{
			"type": "image",
			"version": 69,
			"versionNonce": 1754972145,
			"isDeleted": false,
			"id": "1qKgpmQ2",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"angle": 0,
			"x": -345.1244933896745,
			"y": -172.45297992824743,
			"strokeColor": "#000000",
			"backgroundColor": "transparent",
			"width": 901.1188231699117,
			"height": 304.35139060705626,
			"seed": 10178,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"boundElements": [
				{
					"id": "RKUDd1RLChaGjfhB5mrm8",
					"type": "arrow"
				},
				{
					"id": "0UbCiDM7oGtIX8Q4Owksv",
					"type": "arrow"
				}
			],
			"updated": 1698598110399,
			"link": null,
			"locked": false,
			"status": "pending",
			"fileId": "f00f945039192360a1dcce6c8fd10025bb190243",
			"scale": [
				1,
				1
			]
		},
		{
			"id": "nDxa0sMZ_EWTPeZcNHkTS",
			"type": "arrow",
			"x": -367.28049304501116,
			"y": -259.23099855971606,
			"width": 109.79883409853807,
			"height": 120.35134631045611,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": {
				"type": 2
			},
			"seed": 1667830097,
			"version": 306,
			"versionNonce": 530751441,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698597948472,
			"link": null,
			"locked": false,
			"points": [
				[
					0,
					0
				],
				[
					109.79883409853807,
					120.35134631045611
				]
			],
			"lastCommittedPoint": null,
			"startBinding": {
				"elementId": "qduYRtMJ",
				"focus": 0.29845143934503815,
				"gap": 6.649871468143772
			},
			"endBinding": null,
			"startArrowhead": null,
			"endArrowhead": "arrow"
		},
		{
			"id": "qduYRtMJ",
			"type": "text",
			"x": -554.3073882381427,
			"y": -290.8808700278598,
			"width": 493.07958984375,
			"height": 25,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 2066260337,
			"version": 137,
			"versionNonce": 81024081,
			"isDeleted": false,
			"boundElements": [
				{
					"id": "nDxa0sMZ_EWTPeZcNHkTS",
					"type": "arrow"
				}
			],
			"updated": 1698597944536,
			"link": null,
			"locked": false,
			"text": "Set of target actions for all agents at time t=1",
			"rawText": "Set of target actions for all agents at time t=1",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 17,
			"containerId": null,
			"originalText": "Set of target actions for all agents at time t=1",
			"lineHeight": 1.25,
			"isFrameName": false
		},
		{
			"id": "xlXDNNLfzQPI4aF8yfQVi",
			"type": "arrow",
			"x": 259.5755018322285,
			"y": -258.05330761527415,
			"width": 32.64233903704178,
			"height": 97.07407249794977,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": {
				"type": 2
			},
			"seed": 1388346801,
			"version": 95,
			"versionNonce": 587717073,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698597957104,
			"link": null,
			"locked": false,
			"points": [
				[
					0,
					0
				],
				[
					-32.64233903704178,
					97.07407249794977
				]
			],
			"lastCommittedPoint": null,
			"startBinding": null,
			"endBinding": null,
			"startArrowhead": null,
			"endArrowhead": "arrow"
		},
		{
			"id": "fFaV0kJ3",
			"type": "text",
			"x": 210.69617931550698,
			"y": -289.9783080670232,
			"width": 265.0197448730469,
			"height": 25,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 1431840703,
			"version": 33,
			"versionNonce": 2142325183,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698597968030,
			"link": null,
			"locked": false,
			"text": "Product over all timesteps",
			"rawText": "Product over all timesteps",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 17,
			"containerId": null,
			"originalText": "Product over all timesteps",
			"lineHeight": 1.25,
			"isFrameName": false
		},
		{
			"id": "RKUDd1RLChaGjfhB5mrm8",
			"type": "arrow",
			"x": 184.62288952796285,
			"y": 211.46428847379468,
			"width": 30.098385633076845,
			"height": 68.54890116390976,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": {
				"type": 2
			},
			"seed": 1347102207,
			"version": 25,
			"versionNonce": 1677695441,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698598110399,
			"link": null,
			"locked": false,
			"points": [
				[
					0,
					0
				],
				[
					-30.098385633076845,
					-68.54890116390976
				]
			],
			"lastCommittedPoint": null,
			"startBinding": {
				"elementId": "XlIc92UV",
				"focus": -0.6420348072782325,
				"gap": 4.638342544611703
			},
			"endBinding": {
				"elementId": "1qKgpmQ2",
				"focus": 0.04361406595587712,
				"gap": 11.01697663107609
			},
			"startArrowhead": null,
			"endArrowhead": "arrow"
		},
		{
			"id": "XlIc92UV",
			"type": "text",
			"x": 153.18165094193654,
			"y": 216.10263101840638,
			"width": 237.7198028564453,
			"height": 25,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 1579205471,
			"version": 73,
			"versionNonce": 940111729,
			"isDeleted": false,
			"boundElements": [
				{
					"id": "RKUDd1RLChaGjfhB5mrm8",
					"type": "arrow"
				}
			],
			"updated": 1698597982634,
			"link": null,
			"locked": false,
			"text": "Product over all agents",
			"rawText": "Product over all agents",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 17,
			"containerId": null,
			"originalText": "Product over all agents",
			"lineHeight": 1.25,
			"isFrameName": false
		},
		{
			"id": "WPG0RXul700vT2zHhNhRb",
			"type": "arrow",
			"x": -424.6520887571527,
			"y": 169.24155658891095,
			"width": 210.92778666283255,
			"height": 72.31851083372669,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": {
				"type": 2
			},
			"seed": 666558801,
			"version": 177,
			"versionNonce": 150842591,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698598024354,
			"link": null,
			"locked": false,
			"points": [
				[
					0,
					0
				],
				[
					210.92778666283255,
					-72.31851083372669
				]
			],
			"lastCommittedPoint": null,
			"startBinding": {
				"elementId": "btfe74IM",
				"focus": -0.5571432552875826,
				"gap": 4.254071396222571
			},
			"endBinding": null,
			"startArrowhead": null,
			"endArrowhead": "arrow"
		},
		{
			"id": "btfe74IM",
			"type": "text",
			"x": -575.7119953585532,
			"y": 173.49562798513352,
			"width": 480.339599609375,
			"height": 50,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 1626794161,
			"version": 273,
			"versionNonce": 2018693809,
			"isDeleted": false,
			"boundElements": [
				{
					"id": "WPG0RXul700vT2zHhNhRb",
					"type": "arrow"
				}
			],
			"updated": 1698598024353,
			"link": null,
			"locked": false,
			"text": "Probability of agent states at time t given\nall previous agent states and scene features S",
			"rawText": "Probability of agent states at time t given\nall previous agent states and scene features S",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 42,
			"containerId": null,
			"originalText": "Probability of agent states at time t given\nall previous agent states and scene features S",
			"lineHeight": 1.25,
			"isFrameName": false
		},
		{
			"id": "0UbCiDM7oGtIX8Q4Owksv",
			"type": "arrow",
			"x": -473.2764022366768,
			"y": -96.80195220155656,
			"width": 110.68382436088217,
			"height": 1.0345306541549917,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": {
				"type": 2
			},
			"seed": 958675359,
			"version": 50,
			"versionNonce": 280782783,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698598110400,
			"link": null,
			"locked": false,
			"points": [
				[
					0,
					0
				],
				[
					110.68382436088217,
					1.0345306541549917
				]
			],
			"lastCommittedPoint": null,
			"startBinding": null,
			"endBinding": {
				"elementId": "1qKgpmQ2",
				"focus": 0.4547413723330961,
				"gap": 17.468084486120006
			},
			"startArrowhead": null,
			"endArrowhead": "arrow"
		},
		{
			"id": "JIfiy6ng",
			"type": "text",
			"x": -801.4069254293947,
			"y": -145.68603282451966,
			"width": 344.0997009277344,
			"height": 50,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 63692991,
			"version": 146,
			"versionNonce": 1846365553,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698598067391,
			"link": null,
			"locked": false,
			"text": "Probability of all agent states at\ntime t=1,2,3,...T given S",
			"rawText": "Probability of all agent states at\ntime t=1,2,3,...T given S",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 42,
			"containerId": null,
			"originalText": "Probability of all agent states at\ntime t=1,2,3,...T given S",
			"lineHeight": 1.25,
			"isFrameName": false
		},
		{
			"id": "ugEOvQTHaVFRMB0mUJtpR",
			"type": "arrow",
			"x": -332.3993118183923,
			"y": -64.13722843742408,
			"width": 107.2195445313572,
			"height": 99.46961678237045,
			"angle": 0,
			"strokeColor": "#2f9e44",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": {
				"type": 2
			},
			"seed": 113538897,
			"version": 212,
			"versionNonce": 304622097,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698598110012,
			"link": null,
			"locked": false,
			"points": [
				[
					0,
					0
				],
				[
					-57.21357815379639,
					60.56734015198526
				],
				[
					50.00596637756081,
					99.46961678237045
				]
			],
			"lastCommittedPoint": null,
			"startBinding": null,
			"endBinding": null,
			"startArrowhead": null,
			"endArrowhead": "arrow"
		},
		{
			"id": "AHiD2Dz2",
			"type": "text",
			"x": -895.5427051150617,
			"y": -51.555930424710766,
			"width": 479.4995422363281,
			"height": 125,
			"angle": 0,
			"strokeColor": "#2f9e44",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 1232541663,
			"version": 396,
			"versionNonce": 588882065,
			"isDeleted": false,
			"boundElements": null,
			"updated": 1698598168727,
			"link": null,
			"locked": false,
			"text": "They treat agent states as conditionally\nindependent at time t given the previous actions\nand scene context so these two probabilities\nare the same.\n",
			"rawText": "They treat agent states as conditionally\nindependent at time t given the previous actions\nand scene context so these two probabilities\nare the same.\n",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 117,
			"containerId": null,
			"originalText": "They treat agent states as conditionally\nindependent at time t given the previous actions\nand scene context so these two probabilities\nare the same.\n",
			"lineHeight": 1.25,
			"isFrameName": false
		},
		{
			"id": "iB4G3zQI",
			"type": "text",
			"x": -216.71305465698242,
			"y": -144.2547607421875,
			"width": 10,
			"height": 25,
			"angle": 0,
			"strokeColor": "#1e1e1e",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 1969324433,
			"version": 2,
			"versionNonce": 748542975,
			"isDeleted": true,
			"boundElements": null,
			"updated": 1698597887026,
			"link": null,
			"locked": false,
			"text": "",
			"rawText": "",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 17,
			"containerId": null,
			"originalText": "",
			"lineHeight": 1.25,
			"isFrameName": false
		},
		{
			"id": "o221o4Zy",
			"type": "text",
			"x": -379.8158988952637,
			"y": -307.9073944091797,
			"width": 10,
			"height": 25,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 1714461105,
			"version": 2,
			"versionNonce": 1174656479,
			"isDeleted": true,
			"boundElements": null,
			"updated": 1698597912133,
			"link": null,
			"locked": false,
			"text": "",
			"rawText": "",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 17,
			"containerId": null,
			"originalText": "",
			"lineHeight": 1.25,
			"isFrameName": false
		},
		{
			"id": "ockPjXcw",
			"type": "text",
			"x": 201.7765684719513,
			"y": 263.86050033564345,
			"width": 10,
			"height": 25,
			"angle": 0,
			"strokeColor": "#1971c2",
			"backgroundColor": "transparent",
			"fillStyle": "hachure",
			"strokeWidth": 1,
			"strokeStyle": "solid",
			"roughness": 1,
			"opacity": 100,
			"groupIds": [],
			"frameId": null,
			"roundness": null,
			"seed": 192920031,
			"version": 2,
			"versionNonce": 953718673,
			"isDeleted": true,
			"boundElements": null,
			"updated": 1698597973774,
			"link": null,
			"locked": false,
			"text": "",
			"rawText": "",
			"fontSize": 20,
			"fontFamily": 1,
			"textAlign": "left",
			"verticalAlign": "top",
			"baseline": 17,
			"containerId": null,
			"originalText": "",
			"lineHeight": 1.25,
			"isFrameName": false
		}
	],
	"appState": {
		"theme": "light",
		"viewBackgroundColor": "#ffffff",
		"currentItemStrokeColor": "#2f9e44",
		"currentItemBackgroundColor": "transparent",
		"currentItemFillStyle": "hachure",
		"currentItemStrokeWidth": 1,
		"currentItemStrokeStyle": "solid",
		"currentItemRoughness": 1,
		"currentItemOpacity": 100,
		"currentItemFontFamily": 1,
		"currentItemFontSize": 20,
		"currentItemTextAlign": "left",
		"currentItemStartArrowhead": null,
		"currentItemEndArrowhead": "arrow",
		"scrollX": 130.82866301792203,
		"scrollY": 521.248170640598,
		"zoom": {
			"value": 0.5644150716066362
		},
		"currentItemRoundness": "round",
		"gridSize": null,
		"currentStrokeOptions": null,
		"previousGridSize": null
	},
	"files": {}
}
```
%%