<script lang="ts">
	import { onMount } from 'svelte';
	import {
		getScalingLaws,
		fitScalingLaw,
		type ScalingLawInfo,
		type FitResponse,
		type CustomScalingLawConfig
	} from '$lib/api';
	import FileUpload from '$lib/components/file-upload.svelte';
	import LawSelector from '$lib/components/law-selector.svelte';
	import FitButton from '$lib/components/fit-button.svelte';
	import ParamsDisplay from '$lib/components/params-display.svelte';
	import LossCharts from '$lib/components/loss-charts.svelte';
	import CustomLawForm from '$lib/components/custom-law-form.svelte';
	import * as Card from '$lib/components/ui/card';

	let scalingLaws = $state<ScalingLawInfo[]>([]);
	let file = $state<File | null>(null);
	let selectedLaw = $state<string | null>(null);
	let customConfig = $state<CustomScalingLawConfig | null>(null);
	let loading = $state(false);
	let result = $state<FitResponse | null>(null);
	let error = $state<string | null>(null);
	let lawsLoading = $state(true);

	let isCustom = $derived(selectedLaw === 'Custom');
	let canFit = $derived(
		file !== null &&
			selectedLaw !== null &&
			!loading &&
			(!isCustom || customConfig !== null)
	);

	onMount(async () => {
		try {
			scalingLaws = await getScalingLaws();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load scaling laws';
		} finally {
			lawsLoading = false;
		}
	});

	async function handleFit() {
		if (!file || !selectedLaw) return;
		if (isCustom && !customConfig) return;

		loading = true;
		error = null;
		result = null;

		try {
			const response = await fitScalingLaw(
				file,
				selectedLaw,
				isCustom ? customConfig ?? undefined : undefined
			);
			if (response.success) {
				result = response;
			} else {
				error = response.error ?? 'Unknown error occurred';
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to fit scaling law';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Scaling Law Fitter</title>
</svelte:head>

<main class="container mx-auto py-8 px-4 max-w-6xl">
	<div class="space-y-8">
		<div class="text-center space-y-2">
			<h1 class="text-3xl font-bold">Scaling Law Fitter</h1>
			<p class="text-muted-foreground">
				Upload your training data to fit scaling law parameters
			</p>
		</div>

		<Card.Root>
			<Card.Header>
				<Card.Title>Configuration</Card.Title>
				<Card.Description>Select your data file and scaling law type</Card.Description>
			</Card.Header>
			<Card.Content class="space-y-6">
				<div class="grid gap-6 md:grid-cols-2">
					<FileUpload bind:file disabled={loading} />
					<LawSelector laws={scalingLaws} bind:selectedLaw disabled={loading || lawsLoading} />
				</div>

				{#if isCustom}
					<CustomLawForm bind:config={customConfig} disabled={loading} />
				{/if}

				<FitButton {loading} disabled={!canFit} onclick={handleFit} />

				{#if error}
					<div class="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
						<p class="text-sm text-destructive">{error}</p>
					</div>
				{/if}
			</Card.Content>
		</Card.Root>

		{#if result?.success && (result.fitted_params || result.custom_fitted_params) && result.original_data}
			<ParamsDisplay
				params={result.fitted_params}
				customParams={result.custom_fitted_params}
				fitLoss={result.fit_loss ?? 0}
			/>

			<LossCharts
				dataPoints={result.original_data}
				curvesByN={result.curves_by_N ?? []}
				curvesByD={result.curves_by_D ?? []}
				curvesByC={result.curves_by_C ?? []}
			/>
		{/if}
	</div>
</main>
