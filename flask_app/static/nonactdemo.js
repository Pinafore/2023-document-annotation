const intro = introJs();

intro.setOptions({
    steps: [
        {
            intro: "This demo will walk you through how to use this tool"
        },
        
        {
            element: '#home',
            intro: 'This takes you to the previous page with instructions'
        },
        {
            element: '#topicblock',
            intro: 'The list of passages is automatically generated by the AI tool and grouped together based on their major themes. The word clusters are derived from these passages and represent the key characteristics of each theme within the list of documents'
        },
        {
            element: '#recommended',
            intro: 'The red text is an AI recommended document to label'
        },
        {
            element: "#sessionTimer",
            intro: 'The timer shows you the elapsed time. Once you reached the time and created a good set of labels, you can finish and take the survey'
        },
        {
            element: "#finish",
            intro: 'Click here to finish the study and take the survey'
        },
        {
            element: '#demo',
            intro: 'Click this button if you want to go through this demo again'
        }
    
    ]
})

// const hasRunIntro = localStorage.getItem("hasRunIntro");
// if (hasRunIntro !== "1"){
//     intro.start();
//     localStorage.setItem("hasRunIntro", "1");
// }
document.getElementById("demo").addEventListener('click', function(){
    intro.start();
})